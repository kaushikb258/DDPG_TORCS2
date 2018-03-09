#---------------------------------
# AUTHOR: KAUSHIK BALAKRISHNAN
#---------------------------------

import tensorflow as tf
import numpy as np
#import gym
#from gym import wrappers
import tflearn
import argparse
import pprint as pp
import sys

from replay_buffer import ReplayBuffer

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, noise_option):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.noise_option = noise_option
        self.act_noise  = tf.placeholder(tf.bool)

        # Actor Network
        self.inputs, self.out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)


        
        

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, weights_init='xavier', bias_init='zeros')
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300, weights_init='xavier', bias_init='zeros')
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
      
        # GAUSSIAN NOISE ADDED TO EXPLORE 
        # MULTIPLE OPTIONS AVAILABLE 

        tf_cond = tf.cond(self.act_noise, true_fn = lambda: tf.constant(1.0), false_fn = lambda: tf.constant(0.0))

  
#        if (self.noise_option == 1):
          # OPTION #1
          # ADD N(0,0.2) TO net
#          net = tf.add(net,tf.random_normal(shape=[300], mean=0.0, stddev=0.2, dtype=tf.float32)*tf_cond)

#        elif (self.noise_option == 2):
          # OPTION #2
          # MULTIPLY net WITH (1 + N(0,0.1))
#          net = tf.multiply(net,1.0 + tf.random_normal(shape=[300], mean=0.0, stddev=0.1, dtype=tf.float32)*tf_cond)
          
#        elif (self.noise_option == 3):
          # OPTION #3
          # DROPOUT: SET SOME ACTIVATIONS IN net TO ZERO 
#          net = tflearn.layers.core.dropout (net, keep_prob=1.0-tf.multiply(tf.constant(0.2),tf_cond))
           
#        else:
#          print("wrong entry for noise_option: ", self.noise_option)
#          sys.exit()



        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        steering = tflearn.fully_connected(net, 1, activation='tanh', weights_init=w_init)
        acceleration = tflearn.fully_connected(net, 1, activation='sigmoid', weights_init=w_init) 
        brake = tflearn.fully_connected(net, 1, activation='sigmoid', weights_init=w_init)     
      
        out = tflearn.layers.merge_ops.merge ([steering, acceleration, brake], mode='concat', axis=1)

        return inputs, out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.act_noise: False
        })

    def predict(self, inputs, noise=False):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs, self.act_noise: noise
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs, self.act_noise: False
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        #self.predicted_q_value = tf.placeholder(tf.float32, [None, self.a_dim])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, weights_init='xavier', bias_init='zeros')
        tflearn.add_weights_regularizer(net, 'L2', weight_decay=0.001)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300, weights_init='xavier', bias_init='zeros')
        t2 = tflearn.fully_connected(action, 300, weights_init='xavier', bias_init='zeros')
        
        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        #out = tflearn.fully_connected(net, self.a_dim, weights_init=w_init)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

