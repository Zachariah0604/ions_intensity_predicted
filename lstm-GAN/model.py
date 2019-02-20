import numpy as np

import sys 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dirs = os.path.join( os.path.dirname(__file__),'..')
os.sys.path.append(os.path.join( os.path.dirname(__file__), '..'))
from get_data import *
from pearson import *
from get_batch import *
from lstm import *
import tensorflow as tf

class GRAN(object):
    def __init__(self, 
                 gan_input_size_G,gan_input_size_D,
                 gan_hidden_size,gan_lr,
                 lstm_time_steps,lstm_input_size,lstm_output_size,
                 lstm_layer_num,lstm_cell_size,lstm_lr):
        self.keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')

       
        self.gan_input_size_G=gan_input_size_G
        self.gan_input_size_D=gan_input_size_D
        self.gan_hidden_size = gan_hidden_size
        self.gan_learning_rate = gan_lr

        self.lstm_time_steps=lstm_time_steps
        self.lstm_input_size=lstm_input_size
        self.lstm_output_size=lstm_output_size
        self.lstm_layer_num=lstm_layer_num
        self.lstm_cell_size=lstm_cell_size
        self.lstm_learning_rate=lstm_lr
       
        
        self._create_gan_model()


    def generator(self):
        with tf.name_scope('lstm_inputs'):
            self.lstm_X = tf.placeholder(tf.float32, [None, self.lstm_time_steps, self.lstm_input_size], name='lstm_X')
            self.lstm_y = tf.placeholder(tf.float32, [None, self.lstm_time_steps, self.lstm_output_size], name='lstm_y')
        with tf.variable_scope('lstm_in_hidden'):
            self.add_lstm_input_layer()
        with tf.variable_scope('lstm_cell'):
            self.add_lstm_cell()
        with tf.variable_scope('lstm_out_hidden'):
            self.add_lstm_output_layer()
        #with tf.name_scope('lstm_loss'):
        #    self.compute_lstm_cost()
        #with tf.name_scope('lstm_train'):
        #    self.lstm_train_op = tf.train.AdamOptimizer(self.lstm_learning_rate).minimize(self.lstm_loss)
        return self.lstm_pred

    def add_lstm_input_layer(self):
        l_in_x = tf.reshape(self.lstm_X, [-1, self.lstm_input_size], name='2_2D')  
        Ws_in = self._weight_variable([self.lstm_input_size, self.lstm_cell_size])
        bs_in = self._bias_variable([self.lstm_cell_size,])
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        self.lstm_l_in_y = tf.reshape(l_in_y, [-1, self.lstm_time_steps, self.lstm_cell_size], name='2_3D')

    def add_lstm_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_cell_size, forget_bias=0.0, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.lstm_layer_num, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.lstm_cell_init_state = mlstm_cell.zero_state(tf.shape(self.batch_size)[0], dtype=tf.float32)
        self.lstm_cell_outputs, self.lstm_cell_final_state = tf.nn.dynamic_rnn(
            mlstm_cell, self.lstm_l_in_y, dtype=tf.float32,sequence_length=self.batch_size, time_major=False)

    def add_lstm_output_layer(self):
        l_out_x = tf.reshape(self.lstm_cell_outputs, [-1, self.lstm_cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.lstm_cell_size, self.lstm_output_size])
      
        bs_out = self._bias_variable([self.lstm_output_size, ])
      
        with tf.name_scope('Wx_plus_b'):
            self.lstm_pred = tf.sigmoid(tf.matmul(l_out_x, Ws_out) + bs_out)
        tf.add_to_collection('pred_network', self.lstm_pred )
        

    def compute_lstm_cost(self,real_data,fake_data):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(fake_data, [-1], name='reshape_pred')],
            [tf.reshape(real_data, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.lstm_time_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        _loss = tf.div(
            tf.reduce_sum(losses, name='losses_sum'),
            tf.cast(self.batch_size,tf.float32),
            name='average_loss')
        tf.summary.scalar('lstm_loss', _loss)
        return _loss




    
    def optimizer(self,loss, var_list, initial_learning_rate):
        decay = 0.95
        num_decay_steps = 150
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            batch,
            num_decay_steps,
            decay,
            staircase=True
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss,
            global_step=batch,
            var_list=var_list
        )
        return optimizer
    
    def linear(self,input, output_dim, scope=None, stddev=1.0):
        norm = tf.random_normal_initializer(stddev=stddev)
        const = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
            b = tf.get_variable('b', [output_dim], initializer=const)
            return tf.matmul(input, w) + b
        


    def discriminator(self,_input, h_dim):
        h0 = tf.tanh(self.linear(_input, h_dim * 2, 'd0'))
        h1 = tf.tanh(self.linear(h0, h_dim * 2, 'd1'))
        h2 = tf.tanh(self.linear(h1, h_dim * 2, 'd2'))
        h3 = tf.sigmoid(self.linear(h2, 1, 'd3'))
        return h3

    

    def _create_gan_model(self):

        with tf.variable_scope('D_pre'):                
            self.pre_input = tf.placeholder(tf.float32, shape=(None, 1))
            self.pre_labels = tf.placeholder(tf.float32, shape=(None, 1))
            self.D_pre = self.discriminator(self.pre_input, self.gan_hidden_size)
            self.pre_loss = tf.reduce_mean(tf.square(self.D_pre - self.pre_labels))
            self.pre_opt = self.optimizer(self.pre_loss, None, self.gan_learning_rate)

        with tf.variable_scope('Generator'):
            self.G = self.generator()

       
        with tf.variable_scope('Discriminator') as scope:
            self.D_x = tf.placeholder(tf.float32, [None,self.gan_input_size_D],name='D_x')
            self.D1 = self.discriminator(self.D_x, self.gan_hidden_size)
            scope.reuse_variables()
            self.D2 = self.discriminator(self.G, self.gan_hidden_size)

        
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = self.compute_lstm_cost(self.D1,self.D2) 
        #self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        tf.add_to_collection('loss_d', self.loss_d ) 
        #self.loss_g = tf.reduce_mean(-tf.log(self.D2))
        tf.add_to_collection('loss_g', self.loss_g )

        vars = tf.trainable_variables()
        self.d_pre_params = [v for v in vars if v.name.startswith('D_pre/')]
        self.d_params = [v for v in vars if v.name.startswith('Discriminator/')]
        self.g_params = [v for v in vars if v.name.startswith('Generator/')]

        with tf.name_scope('discriminator-train'):
            #self.opt_d=tf.train.AdamOptimizer(self.gan_learning_rate).minimize(self.loss_d)
            self.opt_d = self.optimizer(self.loss_d, self.d_params, self.gan_learning_rate)
        with tf.name_scope('generator-train'):
            #self.opt_g=tf.train.AdamOptimizer(self.gan_learning_rate).minimize(self.loss_g)
            self.opt_g = self.optimizer(self.loss_g, self.g_params, self.gan_learning_rate)


    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='w'):
        #initializer = tf.random_normal_initializer(mean=0, stddev=1.,)
        return tf.get_variable(shape=shape,  name=name)

    def _bias_variable(self, shape, name='b'):
        #initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape)
