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

"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""
import time
import numpy as np
import tensorflow as tf
from Maze import Maze
import matplotlib.pyplot as plt

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = True
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = True  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.0005    # learning rate for actor
LR_C = 0.002     # learning rate for critic

REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=300)
][1]            

# you can try different target replacement strategies
MEMORY_CAPACITY = 300
BATCH_SIZE = 32


env = Maze()
N_F = env.n_features
N_A = env.n_actions


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.a = tf.placeholder(tf.int32, [None,1], "act")
        self.td_error = tf.placeholder(tf.float32, [None,1], "td_error")  # TD_error

        with tf.variable_scope('Actor'):
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
                units=n_actions,    # output units
                activation=None,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='all_act'
            )
            
            self.acts_prob = tf.nn.softmax(all_act, name='acts_prob')

        with tf.variable_scope('exp_v'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=all_act, labels=tf.one_hot(self.a, n_actions))
            #neg_log_prob = tf.reduce_sum(-tf.log(self.acts_prob)*tf.one_hot(self.a, n_actions))
            self.exp_v = tf.reduce_mean(neg_log_prob * self.td_error)
            

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        #s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


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
            #self.a = a
            self.v = self._build_net(self.s, 'eval_net', trainable=True)
            self.v_ = self._build_net(self.s, 'target_net', trainable=False)
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
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
                units=40,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
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
        #s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v_, {self.s: s_})
        
        td_error, _, v = self.sess.run([self.td_error, self.train_op,self.v],
                                          {self.s: s, self.v_: v_, self.r: r})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
            #print('**********************soft_replacement**************************')
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
                print('**********************hard_replacement**************************')
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

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, replacement = REPLACEMENT, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())
M = Memory(MEMORY_CAPACITY, dims=2 * N_F + 1 + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)
    
def run_Maze():
    
    #var = 3
    td_cost = []
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []
        
        global RENDER
        while True:
            if RENDER: env.render()
    
            a = actor.choose_action(s)
            
            #time.sleep(0.05)
    
            s_, r, done= env.step(a)
    
            #if done: r = -0.1
            M.store_transition(s, a, r, s_)
    
            if M.pointer > MEMORY_CAPACITY:
                #var *= .9999    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :N_F]
                b_a = b_M[:, N_F: N_F + 1]
                b_r = b_M[:, -N_F - 1: -N_F]
                b_s_ = b_M[:, -N_F:]
                
                '''
                b_s = b_M[:, :N_F][0]
                b_a = b_M[:, N_F: N_F + 1][0]
                b_r = b_M[:, -N_F - 1: -N_F][0]
                b_s_ = b_M[:, -N_F:][0]
                b_a = int(b_a)
               '''
    
                td_error = critic.learn(b_s, b_r, b_s_)
                # gradient = grad[r + gamma * V(s_) - V(s)]
                actor.learn(b_s, b_a, td_error)
                # true_gradient = grad[logPi(s,a) * td_error]
                td_cost.append(td_error)
                
    
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
'''
    plt.plot(np.arange(len(td_cost)), td_cost)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()
    
'''
env.after(100, run_Maze)
env.mainloop()
