# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:07:55 2018

@author: Ivy_y
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:16:20 2018

@author: Ivy_y
"""

import time
import numpy as np
import tensorflow as tf
from Maze import Maze


np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = True
MAX_EPISODE = 10000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = True  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0004     # learning rate for critic

REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=300, rep_iter_c=200)
][1]            

# you can try different target replacement strategies
MEMORY_CAPACITY = 300
BATCH_SIZE = 32


env = Maze()
N_F = env.n_features
N_A = env.n_actions


class Actor(object):
    def __init__(self, sess, n_features, n_actions, replacement, lr=0.001, epsilon = 0.9):
        self.sess = sess
        self.replacement = replacement

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.a = tf.placeholder(tf.int32, [None,1], "act")
        self.td_error = tf.placeholder(tf.float32, [None,1], "td_error")
        self.n_actions = n_actions
        self.epsilon = epsilon
        
        with tf.variable_scope('Actor'):
            self.all_act, _ = self._build_net(self.s, 'eval_net', trainable=True)
            _, self.acts_prob= self._build_net(self.s, 'target_net', trainable=False)
        
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')


        with tf.variable_scope('exp_v'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.all_act, labels=tf.one_hot(self.a, n_actions))
            self.exp_v = tf.reduce_mean(neg_log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.exp_v)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]
            
            
    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=40,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            all_act = tf.layers.dense(
                inputs=l1,
                units=self.n_actions,    # output units
                activation=None,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='all_act'
            )
            
            acts_prob = tf.nn.softmax(all_act, name='acts_prob')
            
        return all_act, acts_prob

    def learn(self, s, a, td):

        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
                print('******************actor_hard_replacement***********************')
            self.t_replace_counter += 1


    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        if np.random.uniform() < self.epsilon:
            action = probs[0].argmax()
            
        else:
            action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return action   # return a int


class Critic(object):
    def __init__(self, sess, n_features, replacement, lr=0.01):
        self.sess = sess
        #self.n_features = n_features

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        #self.s_ = tf.placeholder(tf.float32, [None, n_features], "state_")
        #self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.replacement = replacement


        with tf.variable_scope('Critic'):
            self.v = self._build_net(self.s, 'eval_net', trainable=True)
            self.v_ = self._build_net(self.s, 'target_net', trainable=False)
            
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]


    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(
                inputs=s,
                units=40,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
        return v
    
    
    def learn(self, s, r, s_):

        v_ = self.sess.run(self.v_, {self.s: s_})
        
        td_error, _, v = self.sess.run([self.td_error, self.train_op,self.v],
                                          {self.s: s, self.v_: v_, self.r: r})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
            #print('**********************soft_replacement**************************')
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
                print('******************critic_hard_replacement***********************')
            self.t_replace_counter += 1
        
        return td_error
    
        


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]
    

        

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, replacement = REPLACEMENT, lr=LR_A)
critic = Critic(sess, n_features=N_F, replacement = REPLACEMENT, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())
M = Memory(MEMORY_CAPACITY, dims=2 * N_F + 1 + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)
    
def run_Maze():
    
    td_cost = []
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []
        
        global RENDER
        while True:
            if RENDER: env.render()
    
            a = actor.choose_action(s)
            
            #time.sleep(0.1)
    
            s_, r, done= env.step(a)
    
            M.store_transition(s, a, r, s_)
    
            if M.pointer > MEMORY_CAPACITY:

                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :N_F]
                b_a = b_M[:, N_F: N_F + 1]
                b_r = b_M[:, -N_F - 1: -N_F]
                b_s_ = b_M[:, -N_F:]
                

                td_error = critic.learn(b_s, b_r, b_s_)
                # gradient = grad[r + gamma * V(s_) - V(s)]
                actor.learn(b_s, b_a, td_error)
                # true_gradient = grad[logPi(s,a) * td_error]
                td_cost.append(td_error)
                #tf.scalar_summary('loss',td_error)
                
    
            s = s_
            t += 1
            track_r.append(r)
    
            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)
    
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", running_reward)
                break

env.after(100, run_Maze)
env.mainloop()
