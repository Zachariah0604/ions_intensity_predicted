import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys

import utils as uti
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn, rnn_cell 
#import torch
#import torch.nn as nn
#import torch.utils.data as td
#from torch.autograd import Variable
#from torch import optim
#import torch.nn.functional as F
import random
#class RNN(nn.Module):
#    def __init__(self,input_size,hidden_size,num_layers,num_classes):
#        super(RNN,self).__init__()
#      
#        self.hidden_size=hidden_size
#        self.num_layers=num_layers
#        self.lstm = nn.Sequential(
#             
#             nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#             
#                )
#        #self.rnn=nn.RNN(
#        #    input_size,
#        #    hidden_size,
#        #    num_layers,
#        #    batch_first=True
#        #    )
#        self.fc=nn.Linear(hidden_size,num_classes)
#    
#    def forward(self,x):
#       
#        r_out,(h_n,h_c)= self.lstm(x)
#       
#        outs = []    
#        for time_step in range(r_out.size(1)):    
#            outs.append(self.fc(r_out[:, time_step, :]))
#        
#        return torch.stack(outs, dim=1)
#
#class BiLSTM(nn.Module):
#    def __init__(self,args):
#        super(BiLSTM,self).__init__()
#
#        self.hidden_size=args.hidden_size
#        self.num_layers=args.num_layers
#        self.num_classes=args.num_classes
#        self.input_size=args.input_size
#        self.drop=nn.Dropout(0.3)
#        self.relu=nn.ReLU(True)
#        self.sigmoid=nn.Sigmoid()
#        self.lstm=nn.LSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True,bidirectional=True)
#
#
#        self.fc=nn.Linear(self.num_layers*self.hidden_size,self.num_classes)
#
#
#    def init_hidden(self,batch_size):
#        h_n=torch.zeros(self.num_layers*self.num_layers,batch_size,self.hidden_size)
#        h_c=torch.zeros(self.num_layers*self.num_layers,batch_size,self.hidden_size)
#
#        h_n=Variable(h_n).cuda()
#        h_c=Variable(h_c).cuda()
#        return (h_n,h_c)
#
#    def forward(self,X,seq_length):
#        batch_size=X.size(0)
#        self.hidden=self.init_hidden(batch_size)
#        X=self.relu(X)
#        X=torch.nn.utils.rnn.pack_padded_sequence(X,seq_length,batch_first=True)
#        output,self.hidden=self.lstm(X)
#        output,_=torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
#        outs=self.fc(output)
#        return outs
#
#class Encoder(nn.Module):
#    def __init__(self,args):
#        super(Encoder, self).__init__()
#        self.hidden_size=args.hidden_size
#        self.num_layers=args.num_layers
#        self.output_dim=args.output_dim
#        self.input_dim=args.input_dim
#        self.drop=nn.Dropout(args.encoder_dropout)
#        self.rnn=nn.LSTM(self.input_dim,self.hidden_size,self.num_layers,dropout=args.encoder_dropout)
#
#    def forward(self, input,sequence_length):
#        batch_size=input.size(0)
#        #X=self.relu(X)
#        input=torch.nn.utils.rnn.pack_padded_sequence(input,sequence_length)
#        output,hidden=self.rnn(input)
#        
#        output,_=torch.nn.utils.rnn.pad_packed_sequence(output)
#        
#        return output,hidden
#
#class Attention(nn.Module): 
#    def __init__(self, hidden_dim): 
#        super(Attention, self).__init__() 
#        self.hidden_dim = hidden_dim 
#        self.attn = nn.Linear(self.hidden_dim * 3, hidden_dim) 
#        self.v = nn.Parameter(torch.zeros(hidden_dim)) 
#        #self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0))) 
#    def forward(self, hidden, encoder_outputs): 
#        max_len = encoder_outputs.size(0) 
#        h = hidden[0][-1].repeat(max_len, 1, 1)
#        
#        attn_energies = self.score(h, encoder_outputs)
#        return attn_energies
#    def score(self, hidden, encoder_outputs): 
#       
#        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) 
#        energy = energy.permute(1, 2, 0)
#        v = self.v.repeat(encoder_outputs.size(1), 1).unsqueeze(1)
#        energy = torch.bmm(v, energy) 
#        return energy.squeeze(1)
#class Decoder(nn.Module): 
#    def __init__(self,args): 
#        super().__init__()
#        self.input_dim=args.input_dim
#        self.hidden_size = args.hidden_size 
#        self.output_dim = args.output_dim 
#        self.num_layers = args.num_layers 
#        self.dropout = nn.Dropout(args.decoder_dropout) 
#        self.attention = Attention(self.hidden_size) 
#
#        self.decoder_rnn = nn.LSTM(self.output_dim + 2*self.hidden_size, self.hidden_size, num_layers=self.num_layers,bidirectional=True,dropout=args.decoder_dropout) 
#        self.out = nn.Linear(self.output_dim + self.hidden_size * 3, self.output_dim) 
#        
#    
#    def forward(self, input, hidden, encoder_outputs): 
#        decoder_input = input.unsqueeze(0)
#        #with attention
#        attn_weight = self.attention(hidden, encoder_outputs)
#        ions = attn_weight.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1) 
#        decoder_input_con = torch.cat((decoder_input, ions), dim=2)
#        decoder_input_con = F.relu(decoder_input_con)
#        _,decoder_hidden = self.decoder_rnn(decoder_input_con, hidden)
#        
#        output = torch.cat((decoder_input.squeeze(0), decoder_hidden[0][-1], ions.squeeze(0)), dim=1)
#      
#        output=F.relu(self.out(output))
#       
#        return output, decoder_hidden, attn_weight
#class pep2inten_py(nn.Module): 
#    def __init__(self, encoder, decoder,teacher_forcing_ratio=0.2): 
#        super().__init__() 
#        self.encoder = encoder 
#        self.decoder = decoder 
#       
#        self.teacher_forcing_ratio = teacher_forcing_ratio 
#    def forward(self, X, y, sequence_length):
#        X = X.permute(1, 0, 2)
#        y = y.permute(1,0,2)
#        
#        batch_size = X.size(1)
#        max_len = X.size(0)
#        output_size = self.decoder.output_dim
#        outputs =Variable(torch.zeros(max_len,batch_size,output_size).cuda())
#        encoder_outputs, hidden = self.encoder(X, sequence_length) 
#        
#
#        decoder_input = Variable(torch.FloatTensor(torch.zeros((batch_size,output_size))).cuda())
#
#        for t in range(0, max_len): 
#            
#            decoder_output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs) 
#            
#            outputs[t] = decoder_output 
#            teacher_force = random.random() < self.teacher_forcing_ratio
#            decoder_input = decoder_output
#            #decoder_input = (y[t] if teacher_force else decoder_output) 
#        return outputs.permute(1,0,2)
#
#    def predict(self, X,  sequence_length, start_ix=1): 
#        X = X.permute(1, 0, 2)
#        
#        
#        batch_size = X.size(1)
#        max_len = X.size(0)
#        output_size = self.decoder.output_dim
#
#        outputs =Variable(torch.zeros(max_len,batch_size,output_size).cuda())
#       
#        encoder_outputs, hidden = self.encoder(X, sequence_length) 
#
#        decoder_input = Variable(torch.FloatTensor(torch.zeros((batch_size,output_size))).cuda())
#        attn_weights = Variable(torch.zeros((max_len, batch_size, max_len)))
#        
#        for t in range(0, max_len):
#            
#            decoder_output, hidden, attn_weight = self.decoder(decoder_input, hidden, encoder_outputs)
#            
#            outputs[t] =decoder_output
#            decoder_input = decoder_output
#            attn_weights[t] = attn_weight 
#        return outputs.permute(1,0,2), attn_weights
##############################################################



#############################################################
class seq2seq(object):
    def __init__(self, args):
        self.input_dim = args.input_dim
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_keep_prob=args.output_keep_prob
        self.output_dim=args.output_dim
        self.learning_rate=args.learning_rate
        self.is_inference = tf.placeholder(tf.bool,name='is_inference')
        self.max_time=tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
        #self.max_time=40
        #self.batch_size=50
        self.sequence_length = tf.placeholder(tf.int32, [None],name='sequence_length')
        self.encoder_inputs = tf.placeholder(tf.float32, shape=[None,None,self.input_dim], name="encoder_inputs") 
        self.decoder_inputs = tf.placeholder(tf.float32, shape=[None,None,self.output_dim], name="decoder_inputs") 
        self.decoder_targets = tf.placeholder(tf.float32, [None,None,self.output_dim],name='decoder_targets')
        self.seq2seq_model()

    def encoder(self):
        with tf.variable_scope("encoder",reuse=tf.AUTO_REUSE) as encoder_scope:
            encoder_inputs_2d=tf.reshape(self.encoder_inputs,[self.batch_size*self.max_time,self.input_dim])
            encoder_cell_inputs=tf.layers.dense(inputs=encoder_inputs_2d,units=self.hidden_size,activation=tf.nn.relu)
            encoder_cell_inputs_3d=tf.reshape(encoder_cell_inputs,[self.batch_size,self.max_time,self.hidden_size])

            
            encoder_fw_cells = []
            encoder_bw_cells = []
            keep_prob=self.output_keep_prob
            for i in range(self.num_layers):
                with tf.variable_scope('encoder_lstm_{}'.format(i)):
                    cell=tf.contrib.rnn.GLSTMCell(self.hidden_size)
                    #keep_prob+= self.output_keep_prob * ( i*1.0 / float(self.num_layers))
                    #cell=rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=self.output_keep_prob)
                    encoder_fw_cells.append(cell)
                    encoder_bw_cells.append(cell)
            encoder_muti_fw_cell = rnn_cell.MultiRNNCell(encoder_fw_cells)
            encoder_muti_bw_cell = rnn_cell.MultiRNNCell(encoder_bw_cells)

            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_muti_fw_cell,
                                                cell_bw=encoder_muti_bw_cell,
                                                inputs=encoder_cell_inputs_3d,
                                                sequence_length=self.sequence_length,
                                                dtype=tf.float32, time_major=False)

            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            self.encoder_final_state=[]
            for i in range(self.num_layers):
                encoder_final_state_c = tf.concat(
                    (encoder_fw_final_state[i].c, encoder_bw_final_state[i].c), 1)

                encoder_final_state_h = tf.concat(
                    (encoder_fw_final_state[i].h, encoder_bw_final_state[i].h), 1)

                encoder_final_state = LSTMStateTuple(
                    c=encoder_final_state_c,
                    h=encoder_final_state_h
                )
                self.encoder_final_state.append(encoder_final_state) 
            return encoder_outputs,encoder_bw_final_state
    

    
    def decoder(self,encoder_outputs,encoder_states):
       
        batch_size,max_time,_=tf.unstack(tf.shape(encoder_outputs))
        decoder_sequence_length=self.sequence_length
        decoder_max_time=max_time
        ##decder cell
        decoder_cells=[]
        for i in range(self.num_layers):
            with tf.variable_scope('decoder_lstm_{}'.format(i)):
                cell=tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size)
                #cell=rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=self.output_keep_prob) 
                decoder_cells.append(cell)
                
        decoder_cell = rnn_cell.MultiRNNCell(decoder_cells)
        decoder_out_layer = tf.layers.Dense(units = self.output_dim,
                        
                             )
        ### attention
        #attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(
        #    num_units=self.hidden_size, 
        #    memory=encoder_outputs,
        #    memory_sequence_length=self.sequence_length,
        #    normalize=True,
        #    #score_mask_value=0.0
        #    
        #    )
        #attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
        #                        attention_mechanism=attention_mechanism,
        #                      
        #                        alignment_history = True) 

        #decoder_initial_state = attn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            #############################################
        with tf.variable_scope("decoder"):
            
            
            #trainning
          
            #decoder_inputs =tf.concat([tf.fill([self.batch_size,1,self.output_dim], -1.0),self.decoder_targets],axis=1) 
       
            training_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(self.decoder_targets, 
                                               sequence_length=decoder_sequence_length,
                                               sampling_probability=tf.constant(0.5)
                                               )
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
                                  helper=training_helper,
                                  initial_state=self.encoder_final_state,
                                   output_layer=decoder_out_layer
                                  )
         
            self.training_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, maximum_iterations=decoder_max_time,impute_finished=True) 
            #training_final_outputs=tf.slice(self.training_outputs.rnn_output,[0,1,0],[self.batch_size,self.max_time,self.output_dim]) 
        #inference
        with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
            def _sample_fn(decoder_outputs):
                 return decoder_outputs
            def _end_fn(_):
                 return tf.tile([False], [self.batch_size]) 
            inference_helper = tf.contrib.seq2seq.InferenceHelper(
                                sample_fn=lambda outputs:outputs,
                                sample_shape=[self.output_dim], 
                                sample_dtype=tf.float32,
                                start_inputs=tf.fill([self.batch_size,self.output_dim], 0.0),
                                end_fn=lambda sample_ids:False)
            inference_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(tf.zeros_like(self.decoder_targets), 
                                               sequence_length=self.sequence_length, 
                                               sampling_probability=tf.constant(0.5),
                                               seed=2018 )
           
           
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                  helper=inference_helper,
                                  initial_state=self.encoder_final_state,
                                   output_layer=decoder_out_layer
                                  )
         
            self.inference_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, maximum_iterations=decoder_max_time,impute_finished=True)
            #inference_final_outputs=tf.slice(self.inference_outputs.rnn_output,[0,1,0],[self.batch_size,self.max_time,self.output_dim])
        #return training_final_outputs,inference_final_outputs
        return self.training_outputs.rnn_output,self.inference_outputs.rnn_output
    def seq2seq_model(self):
        with tf.variable_scope('pep2inten'): 
            
            encoder_outputs,encoder_states=self.encoder()
           
            training_outputs,inference_outputs=self.decoder(encoder_outputs,encoder_states)

            self.decoder_prediction=tf.reshape(tf.cond(self.is_inference,lambda:inference_outputs,lambda:training_outputs),[self.batch_size*self.max_time,self.output_dim])
            
            #self.decoder_prediction=tf.where(tf.logical_not(self.decoder_prediction<tf.constant(0.0)),self.decoder_prediction,tf.zeros([self.batch_size*self.max_time,self.output_dim]))
            #self.decoder_prediction=tf.where(tf.logical_not(self.decoder_prediction>tf.constant(1.0)),self.decoder_prediction,tf.ones([self.batch_size*self.max_time,self.output_dim]))


            mask=tf.to_float(tf.reshape(tf.sequence_mask(self.sequence_length,self.max_time),[self.batch_size*self.max_time,1]))
            labels=tf.reshape(self.decoder_targets,[self.batch_size*self.max_time,self.output_dim])
            self.loss=tf.losses.absolute_difference(labels,self.decoder_prediction,mask)
            tf.summary.scalar('loss', self.loss)
        
            with tf.variable_scope('tain_op'):
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())
        
    def train(self, sess, max_time,encoder_inputs,decoder_targets,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_targets: decoder_targets,
                    self.sequence_length:sequence_length,
                    self.is_inference:False
                   
                    }
       
        _, loss, summary,aa = sess.run([self.train_op, self.loss, self.summary_op,self.training_outputs], feed_dict=feed_dict)
        #if i==0:
        #    with open('data/train_temp.txt','a') as f:
        #        f.write(str(max_time)+','+str(sequence_length[0])+'\n') 
        #        for line in aa[0][0]:
        #            f.write(str(line)+'\n')
        #        f.write('\n\n')
        #        for line in decoder_targets[0]:
        #            f.write(str(line)+'\n')
        #        f.write('#########################################\n')
        return loss, summary

    def eval(self, sess, max_time,encoder_inputs,decoder_targets,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_targets: decoder_targets,
                    self.sequence_length:sequence_length,
                    self.is_inference:True
                    }
        loss,prediction,aa = sess.run([self.loss,self.decoder_prediction,self.inference_outputs], feed_dict=feed_dict)
        #if i==0:
        #    with open('data/temp.txt','a') as f:
        #        f.write(str(max_time)+','+str(sequence_length[0])+'\n') 
        #        for line in aa[0][0]:
        #            f.write(str(line)+'\n')
        #        f.write('\n\n')
        #        for line in decoder_targets[0]:
        #            f.write(str(line)+'\n')
        #        f.write('#########################################\n')
        return loss,prediction

    def predict(self, sess, max_time,encoder_inputs,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.sequence_length:sequence_length,
                    self.is_inference:True
                    }
        prediction = sess.run(self.decoder_prediction, feed_dict=feed_dict)
        return prediction

class pep2inten(object):
    def __init__(self, args):
        self.input_dim = args.input_dim
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_keep_prob=args.output_keep_prob
        self.output_dim=args.output_dim
        self.learning_rate=args.learning_rate
        self.l1_lamda=args.l1_lamda
        #self.l2_lamda=args.l2_lamda
        self.is_inference = tf.placeholder(tf.bool,name='is_inference')
        self.max_time=tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
        self.sequence_length = tf.placeholder(tf.int32, [None],name='sequence_length')
        self.encoder_inputs = tf.placeholder(tf.float32, shape=[None,None,self.input_dim], name="encoder_inputs") 
        self.decoder_targets = tf.placeholder(tf.float32, [None,None,self.output_dim],name='decoder_targets')
        self.pep2inten()
    def attention(self,atten_inputs, attention_size):
        with tf.variable_scope("attention"):
            inputs_hidden_size=6*self.hidden_size
            # Attention mechanism
            attention_w = tf.get_variable(shape=[inputs_hidden_size, attention_size], name='attention_w')
            attention_b = tf.get_variable(shape=[attention_size], name='attention_b')

            
            u = tf.nn.tanh(tf.matmul(tf.reshape(atten_inputs, [-1, inputs_hidden_size]), attention_w) + attention_b)
            u_w =tf.Variable(tf.random_normal([attention_size,1], stddev=0.1))
            atten_score = tf.reshape(tf.matmul(u, u_w), [self.batch_size, self.max_time, 1])
            #mask
            mask = tf.cast(tf.sequence_mask(self.sequence_length,self.max_time), tf.float32)
            mask = tf.expand_dims(mask, 2)
            mask_atten_score=atten_score - (1 - mask) * 1e12

            alpha =tf.nn.softmax(mask_atten_score)
            atten_outputs = atten_inputs * alpha
            
            return atten_outputs
    def get_cell(self,hidden_size,num_layers):
        cells = []
        keep_prob=self.output_keep_prob
        for i in range(num_layers):
            with tf.variable_scope('cell_{}'.format(i)):
                cell=tf.contrib.rnn.LSTMCell(hidden_size,use_peepholes=True)
               
                #keep_prob+= self.output_keep_prob * ( i*1.0 / float(self.num_layers))
                #cell=rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=self.output_keep_prob)
                cells.append(cell)
        muti_cells = rnn_cell.MultiRNNCell(cells)
        return muti_cells
    def encoder(self):
        with tf.variable_scope("encoder"):

            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.get_cell(self.hidden_size,self.num_layers),
                                                cell_bw=self.get_cell(self.hidden_size,self.num_layers),
                                                inputs=self.encoder_inputs,
                                                sequence_length=self.sequence_length,
                                                dtype=tf.float32, time_major=False)
            encoder_outputs=tf.concat([encoder_fw_outputs, encoder_bw_outputs],axis=2)

            encoder_final_state_c = tf.concat((encoder_fw_final_state[self.num_layers-1].c, encoder_bw_final_state[self.num_layers-1].c), 1)
            encoder_final_state_h = tf.concat((encoder_fw_final_state[self.num_layers-1].h, encoder_bw_final_state[self.num_layers-1].h), 1)
            encoder_final_state = LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h)
            hidden_state=tf.concat(encoder_final_state,axis=1)
           
            return encoder_outputs,hidden_state
    def decoder(self,encoder_outputs,hidden_state):
        with tf.variable_scope("decoder"):
            decoder_hidden_size=self.hidden_size//4
            
            hidden_state=tf.tile(tf.expand_dims(hidden_state,1),multiples=[1,self.max_time,1])
            atten_inputs=tf.concat([encoder_outputs,hidden_state],axis=2)
            atten_outputs=self.attention(atten_inputs,self.hidden_size)

            decoder_initial_inputs=tf.concat([atten_outputs,hidden_state],axis=2)
            
          
          
            decoder_outputs,_=tf.nn.dynamic_rnn(cell=self.get_cell(decoder_hidden_size,self.num_layers),
                                                inputs=decoder_initial_inputs,
                                                sequence_length=self.sequence_length,
                                                dtype=tf.float32, time_major=False)

           
            outputs=tf.reshape(decoder_outputs,[self.batch_size*self.max_time,decoder_hidden_size])
            finally_outputs=tf.layers.dense(inputs=outputs,units=self.output_dim,kernel_regularizer=tf.contrib.layers.l1_regularizer(self.l1_lamda))
            return  finally_outputs
   
    def pep2inten(self):
       
        with tf.variable_scope('pep2inten'): 
            encoder_outputs,hidden_state=self.encoder()
            self.decoder_prediction=self.decoder(encoder_outputs,hidden_state)

        with tf.variable_scope('loss'):
            mask=tf.to_float(tf.reshape(tf.sequence_mask(self.sequence_length,self.max_time),[self.batch_size*self.max_time,1]))
            labels=tf.reshape(self.decoder_targets,[self.batch_size*self.max_time,self.output_dim])
           
            self.loss=tf.losses.mean_squared_error(labels,self.decoder_prediction,weights=mask)
           
            tf.summary.scalar('loss', self.loss)
            tf.add_to_collection('loss', self.loss)

        with tf.variable_scope('tain_op'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)
            tf.add_to_collection('tain_op', self.train_op)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables()) 
   
    def train(self, sess, max_time,encoder_inputs,decoder_targets,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_targets: decoder_targets,
                    self.sequence_length:sequence_length,
                    self.is_inference:False
                   
                    }
       
        _, loss, summary= sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        
        return loss, summary

    def eval(self, sess, max_time,encoder_inputs,decoder_targets,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_targets: decoder_targets,
                    self.sequence_length:sequence_length,
                    self.is_inference:True
                    }
        loss,prediction = sess.run([self.loss,self.decoder_prediction], feed_dict=feed_dict)
        
        return loss,prediction
    def predict(self, sess, max_time,encoder_inputs,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.sequence_length:sequence_length,
                    self.is_inference:True
                    }
        prediction = sess.run(self.decoder_prediction, feed_dict=feed_dict)
        
        return prediction

 
class W_SEQ_GAN(object):
    def __init__(self,args):
        self.input_dim = args.input_dim
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_keep_prob=args.output_keep_prob
        self.output_dim=args.output_dim
        self.learning_rate=args.learning_rate
        self.max_time=tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
        self.sequence_length = tf.placeholder(tf.int32, [None],name='sequence_length')
        self.gen_inputs = tf.placeholder(tf.float32, shape=[None,None,self.input_dim], name="gen_inputs") 
        self.targets_intensity = tf.placeholder(tf.float32, [None,None,self.output_dim],name='targets_intensity')
              
        self._create_gan_model()

    def get_cell(self,hidden_size):
        fw_cells = []
        bw_cells = []
        keep_prob=self.output_keep_prob
        for i in range(self.num_layers):
            with tf.variable_scope('lstm_{}'.format(i)):
                cell=tf.contrib.rnn.GLSTMCell(hidden_size)
                #keep_prob+= self.output_keep_prob * ( i*1.0 / float(self.num_layers))
                cell=rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=self.output_keep_prob)
                fw_cells.append(cell)
                bw_cells.append(cell)
        muti_fw_cell = rnn_cell.MultiRNNCell(fw_cells)
        muti_bw_cell = rnn_cell.MultiRNNCell(bw_cells)
        return muti_fw_cell,muti_bw_cell
    def generator(self):

        with tf.variable_scope("Generator",reuse=tf.AUTO_REUSE):
            gen_inputs_2d=tf.reshape(self.gen_inputs,[self.batch_size*self.max_time,self.input_dim])
            gen_cell_inputs=tf.layers.dense(inputs=gen_inputs_2d,units=self.hidden_size,activation=tf.nn.relu)
            gen_cell_inputs_3d=tf.reshape(gen_cell_inputs,[self.batch_size,self.max_time,self.hidden_size])

            
            
            gen_muti_fw_cell,gen_muti_bw_cell=self.get_cell(self.hidden_size)
            (gen_fw_outputs, gen_bw_outputs), (gen_fw_final_state, gen_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=gen_muti_fw_cell,
                                                cell_bw=gen_muti_bw_cell,
                                                inputs=gen_cell_inputs_3d,
                                                sequence_length=self.sequence_length,
                                                dtype=tf.float32, time_major=False)

            gen_outputs = tf.concat((gen_fw_outputs, gen_bw_outputs), 2)

            
            gen_prediction=tf.layers.dense(inputs=tf.reshape(gen_outputs,[self.batch_size*self.max_time,self.hidden_size*2]),units=self.output_dim)

            return gen_prediction
    

    


    def discriminator(self,disc_inputs,is_real):
        with tf.variable_scope('Discriminator',reuse=tf.AUTO_REUSE):
            disc_hidden_size=self.hidden_size//4
            cell_inputs=tf.layers.dense(inputs=tf.reshape(disc_inputs,[self.batch_size*self.max_time,self.output_dim]),units=disc_hidden_size)
            cell_inputs=tf.reshape(cell_inputs,[self.batch_size,self.max_time,disc_hidden_size])

            disc_muti_fw_cell,disc_muti_bw_cell=self.get_cell(disc_hidden_size)

            #(disc_fw_outputs, disc_bw_outputs), (disc_fw_final_state, disc_bw_final_state) = \
            #    tf.nn.bidirectional_dynamic_rnn(cell_fw=disc_muti_fw_cell,
            #                                    cell_bw=disc_muti_bw_cell,
            #                                    inputs=cell_inputs,
            #                                    sequence_length=self.sequence_length,
            #                                    dtype=tf.float32, time_major=False)

            #disc_outputs = tf.concat((disc_fw_outputs, disc_bw_outputs), 2)
            disc_outputs,disc_final_state=tf.nn.dynamic_rnn(cell=disc_muti_fw_cell,inputs=cell_inputs,sequence_length=self.sequence_length,dtype=tf.float32,time_major=False)

            disc_outputs=tf.layers.dense(inputs=tf.reshape(disc_outputs,[self.batch_size*self.max_time,disc_hidden_size]),units=1)
            if is_real:
                disc_loss=tf.nn.l2_loss(disc_outputs-tf.ones([self.batch_size*self.max_time,1]))
            else:
                disc_loss=tf.nn.l2_loss(disc_outputs-tf.zeros([self.batch_size*self.max_time,1])) 
            return disc_loss    
    def _create_gan_model(self):
        
        self.discriminator(self.targets_intensity,True)
        self.gen_outputs=self.generator()


        mask=tf.to_float(tf.reshape(tf.sequence_mask(self.sequence_length,self.max_time),[self.batch_size*self.max_time,1]))
            
        targets_intensity=tf.reshape(self.targets_intensity,[self.batch_size*self.max_time,self.output_dim]) 
        self.loss_g = tf.losses.absolute_difference(self.gen_outputs,targets_intensity,mask)+self.discriminator(self.gen_outputs,True)
        self.loss_d = self.discriminator(self.gen_outputs,False)
        #self.loss_d = tf.nn.l2_loss(self.disc_fake-self.disc_real)
        tf.summary.scalar('gen_loss', self.loss_g)
        tf.summary.scalar('disc_loss', self.loss_d) 
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator')

      
        print("Generator params:")
        for var in self.g_params:
            print("\t{}\t{}".format(var.name, var.get_shape()))
        print("Discriminator params:")
        for var in self.d_params:
            print("\t{}\t{}".format(var.name, var.get_shape()))
       
           
        with tf.name_scope('discriminator-train'):
            self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_d, var_list=self.d_params)
            tf.add_to_collection('disc_train_op', self.disc_train_op )
        with tf.name_scope('generator-train'):
            self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_g, var_list=self.g_params)
            

            #tf.add_to_collection('gen_train_op',self.gen_train_op )    
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())
    def train(self, sess, max_time,gen_inputs,targets_intensity,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(gen_inputs),
                    self.gen_inputs: gen_inputs,
                    self.targets_intensity: targets_intensity,
                    self.sequence_length:sequence_length
                   
                    }
       
        _, loss_d,loss_g,summary = sess.run([self.disc_train_op,self.loss_d,self.loss_g, self.summary_op], feed_dict=feed_dict)
        
        return loss_d,loss_g, summary

    def eval(self, sess, max_time,gen_inputs,targets_intensity,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(gen_inputs),
                    self.gen_inputs: gen_inputs,
                    self.targets_intensity: targets_intensity,
                    self.sequence_length:sequence_length
                    }
        prediction,loss_d,loss_g,summary = sess.run([self.gen_outputs,self.loss_d,self.loss_g, self.summary_op], feed_dict=feed_dict)
       
        return loss_d,loss_g,prediction
    def predict(self, sess, max_time,gen_inputs,targets_intensity,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(gen_inputs),
                    self.gen_inputs: gen_inputs,
                    self.sequence_length:sequence_length
                    }
        prediction = sess.run(self.gen_outputs, feed_dict=feed_dict)
        return prediction
 
       