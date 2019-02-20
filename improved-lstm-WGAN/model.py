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
import utils as uti
from tensorflow.python.ops import rnn, rnn_cell 

class WGRAN(object):
    def __init__(self, _lambda,mse_proportion,linear_initialization,
                 gan_input_size_G,gan_input_size_D,
                 D_hidden_size,learning_rate,
                 lstm_time_steps,lstm_input_size,lstm_output_size,
                 lstm_layer_num,lstm_cell_size):
        self.keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
       
        self._lambda=_lambda
        self.mse_proportion=mse_proportion
        if linear_initialization=='None':
            self.linear_initialization=None
        else:
            self.linear_initialization=linear_initialization
               
        self.gan_input_size_G=gan_input_size_G
        self.gan_input_size_D=gan_input_size_D
        self.D_hidden_size = D_hidden_size
        self.learning_rate = learning_rate

        self.lstm_time_steps=lstm_time_steps
        self.lstm_input_size=lstm_input_size
        self.lstm_output_size=lstm_output_size
        self.lstm_layer_num=lstm_layer_num
        self.lstm_cell_size=lstm_cell_size
       
        self._create_gan_model()


    def generator(self):
        
        self.lstm_X = tf.placeholder(tf.float32, [None, self.lstm_time_steps, self.lstm_input_size], name='lstm_X')

        l_in_x = tf.reshape(self.lstm_X, [-1, self.lstm_input_size], name='2_2D')  
        l_in_y=uti._linear('Generator.LSTM_in_layer.Linear',self.lstm_input_size,self.lstm_cell_size,l_in_x,initialization=self.linear_initialization)
        #l_in_y=uti.batch_norm(l_in_y)
        lstm_l_in_y =tf.reshape(l_in_y, [-1, self.lstm_time_steps, self.lstm_cell_size])
        
        with tf.variable_scope('Generator.LSTM_cell'):  
            lstm_cell_outputs,lstm_cell_final_state=self.add_lstm_cell(lstm_l_in_y)
            self.gen_lstm_cell_final_state=lstm_cell_final_state
        
        l_out_x = tf.reshape(lstm_cell_outputs, [-1, self.lstm_cell_size], name='2_2D')
        lstm_pred=uti._linear('Generator.LSTM_out_layer.Linear',self.lstm_cell_size,self.lstm_output_size,l_out_x,initialization=self.linear_initialization)
        #lstm_pred=uti.batch_norm(lstm_pred)
        lstm_pred=tf.nn.relu(lstm_pred)
        tf.add_to_collection('pred_network', lstm_pred )

        return lstm_pred
            
    

    def add_lstm_cell(self,lstm_l_in_y):
        
        lstm_cell = rnn_cell.BasicLSTMCell(self.lstm_cell_size, forget_bias=1.0, state_is_tuple=True)
        
        lstm_cell = rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        mlstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * self.lstm_layer_num, state_is_tuple=True)
        lstm_cell_outputs, lstm_cell_final_state = tf.nn.dynamic_rnn(
            mlstm_cell, lstm_l_in_y, dtype=tf.float32,sequence_length=self.batch_size, time_major=False)
        
        return lstm_cell_outputs, lstm_cell_final_state

    
    

    def ReLULayer(self,name, n_in, n_out, inputs):
        output =  uti._linear(
            name+'.Linear',
            n_in,
            n_out,
            inputs,
            initialization='he'

        )
        #output=uti.batch_norm(output)
        output =tf.nn.relu(output)
        return output


    def discriminator(self,_input):
       
        output = self.ReLULayer('Discriminator.1', 1, self.D_hidden_size, _input)
        output = self.ReLULayer('Discriminator.2', self.D_hidden_size, self.D_hidden_size, output) 
        
        output = self.ReLULayer('Discriminator.3', self.D_hidden_size, self.D_hidden_size, output)
        
        output = uti._linear('Discriminator.4', self.D_hidden_size, 1, output)
        #return output

        #input = self.ReLULayer('Discriminator.1', 1, self.lstm_cell_size, _input)
        #l_in_y=uti._linear('Discriminator.LSTM_in_layer.Linear',self.lstm_output_size,self.lstm_cell_size,_input,initialization=self.linear_initialization)
        #lstm_l_in_y =tf.reshape(l_in_y, [-1, self.lstm_time_steps, self.lstm_cell_size])
        
        #with tf.variable_scope('Discriminator.LSTM_cell'):  
        #    lstm_cell_outputs,lstm_cell_final_state=self.add_lstm_cell(lstm_l_in_y)
        #    self.disc_lstm_cell_final_state=lstm_cell_final_state
        
        #l_out_x = tf.reshape(lstm_cell_outputs, [-1, self.lstm_cell_size], name='2_2D')
        ##l_out_x = self.ReLULayer('Discriminator.2', self.lstm_cell_size, self.lstm_cell_size, l_out_x)
        #output=uti._linear('Discriminator.LSTM_out_layer.Linear',self.lstm_cell_size,self.lstm_output_size,l_out_x,initialization=self.linear_initialization)

        
        return output      
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

    def _create_gan_model(self):
        
        with tf.variable_scope('Generator'):
            self.fake_data = self.generator()
        with tf.variable_scope('Discriminator'):
            self.real_data = tf.placeholder(tf.float32, [None,self.gan_input_size_D],name='D_x')
            self.disc_real = self.discriminator(self.real_data)
           
            self.disc_fake = self.discriminator(self.fake_data)

        self.loss_d = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.loss_g = (1-self.mse_proportion)*(-tf.reduce_mean(self.disc_fake))+self.mse_proportion*self.compute_mse(self.real_data,self.fake_data)

       
        alpha = tf.random_uniform(
        shape=[self.batch_size,1], 
        minval=0.,
        maxval=1.
        )
        differences = self.fake_data - self.real_data
        interpolates = self.real_data + (alpha*differences)
        self.disc_interpolates=self.discriminator(interpolates)
        gradients = tf.gradients(self.disc_interpolates, [interpolates])
        print(gradients)
        
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        self.loss_d += gradient_penalty*self._lambda


        tf.summary.scalar('d_loss', self.loss_d)
        tf.summary.scalar('g_loss', self.loss_g)
        tf.add_to_collection('loss_d', self.loss_d )
        tf.add_to_collection('loss_g', self.loss_g )

        
      
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator')

      
        print("Generator params:")
        for var in self.g_params:
            print("\t{}\t{}".format(var.name, var.get_shape()))
        print("Discriminator params:")
        for var in self.d_params:
            print("\t{}\t{}".format(var.name, var.get_shape()))

        with tf.name_scope('discriminator-train'):
            self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.5,beta2=0.9).minimize(self.loss_d, var_list=self.d_params)
            tf.add_to_collection('disc_train_op', self.disc_train_op )
        with tf.name_scope('generator-train'):
            if(len(self.g_params)>0):
                self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.5,beta2=0.9).minimize(self.loss_g, var_list=self.g_params)
            else:
                self.gen_train_op = tf.no_op()

            tf.add_to_collection('gen_train_op',self.gen_train_op )    
    def compute_mse(self,real,fake):
        losses = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(fake, [-1], name='reshape_pred')],
            [tf.reshape(real, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.lstm_time_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
        )
        lstm_loss = tf.div(
            tf.reduce_sum(losses),
            tf.cast(self.batch_size,tf.float32),
            )
        tf.summary.scalar('lstm_loss', lstm_loss)
        return lstm_loss
    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    
