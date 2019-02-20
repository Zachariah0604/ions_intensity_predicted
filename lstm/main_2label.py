import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import sys 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dirs = os.path.join( os.path.dirname(__file__),'..')
os.sys.path.append(os.path.join( os.path.dirname(__file__), '..'))
from tools.get_data import *
from tools.pearson import *
from tensorflow.python.ops import rnn, rnn_cell 
import argparse
tf.set_random_seed(1)

NUM_LABEL=2
datam=GetData(NUM_LABEL)
pear=CalcPerson(NUM_LABEL)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-train', type=int, default=0,help='1=train, 2=test, 0=train and test')
    parser.add_argument('--num-iter', type=int, default=200,help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=64,help='the number of peptide')
    parser.add_argument('--keep-prob', type=float, default=0.5,help='')
    parser.add_argument('--learning-rate', type=float, default=1e-4,help='')
    parser.add_argument('--input-size', type=int, default=88,help='')
    parser.add_argument('--output-size', type=int, default=2,help='predict ionic strength')
    parser.add_argument('--layer-num', type=int, default=2,help='')
    parser.add_argument('--cell-size', type=int, default=1024,help='')
    return parser.parse_args()

class LSTM(object):
    def __init__(self,  args):
        self.max_time = tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.input_size = args.input_size
        self.output_size =  args.output_size
        self.cell_size =  args.cell_size
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')

        self.layer_num=args.layer_num
        self.learning_rate=args.learning_rate
        self.keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')
        self.seq_length = tf.placeholder(tf.float32,[None],name='seq_length')
        #self.batch_size = batch_size
       
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [None, None, self.input_size], name='X')
            self.y = tf.placeholder(tf.float32, [None, None, self.output_size], name='y')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        #with tf.name_scope('loss'):
        #    self.compute_cost()
        with tf.name_scope('train'):
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,var_list=self._params)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.X, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        
        bs_in = self._bias_variable([self.cell_size,])
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.nn.relu(tf.matmul(l_in_x, Ws_in) + bs_in)
        self.l_in_y = tf.reshape(l_in_y, [self.batch_size, self.max_time, self.cell_size], name='2_3D')

    def add_cell(self):
        #lstm_cell = rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        
        #lstm_cell = rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        #mlstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * self.layer_num, state_is_tuple=True)
        ##with tf.name_scope('initial_state'):
        ##    self.cell_init_state = mlstm_cell.zero_state(tf.shape(self.batch_size)[0], dtype=tf.float32)
        #self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
        #    mlstm_cell, self.l_in_y,dtype=tf.float32 ,sequence_length=self.seq_length, time_major=False)


        lstm_fw_cell =tf.nn.rnn_cell.LSTMCell(self.cell_size)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.cell_size)

        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=self.keep_prob)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=self.keep_prob)
      

        mlstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * self.layer_num, state_is_tuple=True)
        mlstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * self.layer_num, state_is_tuple=True)
        self.seq_length = tf.cast(self.seq_length, tf.int32) 
        #with tf.name_scope('initial_state'):
        #    self.cell_init_state = mlstm_cell.zero_state(tf.shape(self.batch_size)[0], dtype=tf.float32)
        #lstm_inputs=tf.unstack(self.l_in_y, self.max_time, 1)
        (self.output_fw, self.output_bw), self.states = tf.nn.bidirectional_dynamic_rnn(
                                                                mlstm_fw_cell,
                                                                mlstm_bw_cell,
                                                                self.l_in_y,
                                                                sequence_length=self.seq_length,
                                                                dtype=tf.float32 )

    def add_output_layer(self):
        l_out_x = tf.reshape(tf.concat([self.output_fw, self.output_bw],axis=2), [-1, self.cell_size * 2])
        Ws_out = self._weight_variable([self.cell_size*2, self.output_size])
        #tf.summary.histogram('Ws_out',Ws_out)
        bs_out = self._bias_variable([self.output_size, ])
        #tf.summary.histogram('bs_out',bs_out)
        with tf.name_scope('Wx_plus_b'):
            self.pred =tf.matmul(l_out_x, Ws_out) + bs_out
       
        tf.add_to_collection('pred_network', self.pred )

        #huber_loss
        self.loss=tf.losses.huber_loss(labels=tf.reshape(self.y,[-1,self.output_size]),predictions=self.pred,delta=0.3)
        #logcosh_loss
        #self.loss=tf.reduce_mean(tf.log(tf.cosh(self.pred - tf.reshape(self.y,[-1,self.output_size]))))
        

        tf.summary.scalar('loss', self.loss)

    #def compute_cost(self):
    #    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    #        [tf.reshape(self.pred, [-1], name='reshape_pred')],
    #        [tf.reshape(self.y, [-1], name='reshape_target')],
    #        [tf.ones([self.batch_size * self.max_time], dtype=tf.float32)],
    #        average_across_timesteps=True,
    #        softmax_loss_function=self.ms_error,
    #        name='losses'
    #    )
    #    with tf.name_scope('average_loss'):
    #        self.loss = tf.div(
    #            tf.reduce_sum(losses, name='losses_sum'),
    #            tf.cast(self.batch_size,tf.float32),#self.loss=tf.nn.ctc_loss(labels=self.y,inputs=tf.reshape(self.pred,[self.batch_size,self.max_time,self.output_size]),sequence_length=self.seq_length,time_major=False)
    #            name='average_loss')
    #        tf.summary.scalar('loss', self.loss)
    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        #initializer = tf.zeros_initializer()
        return tf.get_variable(shape=shape,  name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape)

def get_batch_peptide(merge_list,_batch_size):
    number_of_peptide=len(merge_list[0])
    batch_peptide=[]
    seq_length=[]
    _batch_number=int(number_of_peptide/_batch_size)
    for i in range(_batch_number):
        batch_peptide.append(merge_list[0][i*_batch_size:(i+1)*_batch_size])
        seq_length.append([])
        for j in range(len(batch_peptide[i])):
            seq_length[len(batch_peptide)-1].append(len(merge_list[0][i*_batch_size+j]))
    if _batch_number*_batch_size < number_of_peptide:
        seq_length.append([])
        batch_peptide.append(merge_list[0][_batch_number*_batch_size:])
        aa=len(batch_peptide[-1])
        bb=len(batch_peptide)
        for k in range(len(batch_peptide[-1])):
            seq_length[len(batch_peptide)-1].append(len(merge_list[0][_batch_number*_batch_size+k]))
    return batch_peptide,_batch_number,seq_length

def padding_data(data,flag,max_ions_number):
    #if flag ==1 :
    #    _ydim=data.shape[1]
    #else:
    #    _ydim=data.shape[0]
    _ydim=data.shape[1]
    dv=max_ions_number-data.shape[0]
    data=data.tolist()
    if dv > 0:
        #if flag ==1:
        #    data.extend(np.zeros((dv,_ydim)).tolist())
        #else:
        #    data.extend(np.zeros((dv,)).tolist())
        data.extend(np.zeros((dv,_ydim)).tolist())
    return data

def train(args):
    model = LSTM(args)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("lstm/lstm-logs", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        
        _,_,train_X,train_y,merge_train_list=datam.get_data('data/data_swedcad_mm/am/train_2label.txt')
        print(str(len(merge_train_list[0]))+' train peptides ,DataShape:('+str(np.array(train_X).shape)+str(np.array(train_y).shape)+')')
        
        batch_peptide,_batch_number,seq_length=get_batch_peptide(merge_train_list,args.batch_size)
        #if len(seq_length[-1]) < args.batch_size:
        #   seq_length[-1].extend([0]*(args.batch_size-len(seq_length[-1])))
        print('..trainning')

        for Iter in range(args.num_iter):

            permutation_batch = np.random.permutation(len(batch_peptide))
            suffled_batch_peptide=np.array(batch_peptide)[permutation_batch].tolist()
            suffled_seq_length=np.array(seq_length)[permutation_batch].tolist()

            for i,(train_piptide_index) in enumerate(suffled_batch_peptide):
                X=[];y=[];
                max_ions_number=max(suffled_seq_length[i])

                permutation_peptide = np.random.permutation(len(train_piptide_index))
                suffled_seq=np.array(suffled_seq_length[i])[permutation_peptide].tolist()
                suffled_train_piptide_index=np.array(train_piptide_index)[permutation_peptide].tolist()

                #padding_pep_num=args.batch_size-len(suffled_train_piptide_index)
                
                for j in range(len(suffled_train_piptide_index)):
                    train_ion_index=datam.get_split_list(suffled_train_piptide_index[j])
                    X.append(padding_data(train_X[np.array(train_ion_index)],1,max_ions_number))
                    y.append(padding_data(train_y[np.array(train_ion_index)],0,max_ions_number))

                #if padding_pep_num >0:
                #   for k in range(padding_pep_num):
                       
                #       X.append(padding_data(np.zeros((2,279)),1,max_ions_number))
                #       y.append(padding_data(np.zeros((2,2)),0,max_ions_number))
               
                feed_dict = {
                        model.X:np.array(X,dtype=np.float32),
                        model.y:np.array(y,dtype=np.float32),
                        model.keep_prob:args.keep_prob,
                        model.seq_length:suffled_seq,
                        model.max_time:max_ions_number,
                        model.batch_size:len(X)
                        
                }
                _, loss, state, pred = sess.run(
                    [model.train_op, model.loss, model.states, model.pred],
                    feed_dict=feed_dict)
                
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, Iter)
            
            print('Iter[%d/%d],loss[%.4f]' % (Iter+1,args.num_iter,round(loss,4)))
        print("SaveModel:",tf.train.Saver().save(sess,'lstm/model2/bilstm_model.ckpt'))
 
        
def MSE(label,pred):
    return tf.losses.huber_loss(labels=tf.reshape(label,[-1,2]),predictions=pred,delta=0.3)
  
def model_predict(_batch_size,_keep_prob):
    print('predicting..')
    test_idx,peptide,test_X,test_y,merge_test_list=datam.get_data('data/data_swedcad_mm/am/test_2label.txt')
    print(str(len(merge_test_list[0]))+' test peptides ,DataShape:('+str(np.array(test_X).shape)+str(np.array(test_y).shape)+')')
    with tf.Session() as session:
        batch_peptide,_batch_number,_seq_length=get_batch_peptide(merge_test_list,_batch_size)
        #if len(_seq_length[-1]) < _batch_size:
        #   _seq_length[-1].extend([0]*(_batch_size-len(_seq_length[-1])))

        saver = tf.train.import_meta_graph('lstm/model2/bilstm_model.ckpt.meta')
        saver.restore(session, tf.train.latest_checkpoint('lstm/model2/'))
        graph=tf.get_default_graph()
        inputs_X=graph.get_operation_by_name("inputs/X").outputs[0]
        batch_size=graph.get_operation_by_name("batch_size").outputs[0]
        keep_prob=graph.get_operation_by_name("keep_prob").outputs[0]
        max_time=graph.get_operation_by_name("max_time").outputs[0]
        seq_length=graph.get_operation_by_name("seq_length").outputs[0]

        pred_y=tf.get_collection("pred_network")[0]
        pred=[]
        mse_list=[]
        for i,(test_piptide_index) in enumerate(batch_peptide):
            X=[];y=[]
            _max_ions_number=max(_seq_length[i])
            #padding_pep_num=_batch_size-len(test_piptide_index)
            for j in range(len(test_piptide_index)):
                test_ion_index=datam.get_split_list(test_piptide_index[j])
                X.append(padding_data(test_X[np.array(test_ion_index)],1,_max_ions_number))
                y.append(padding_data(test_y[np.array(test_ion_index)],0,_max_ions_number))
            #if padding_pep_num >0:
            #    for k in range(padding_pep_num):
            #       X.append(padding_data(np.zeros((2,88)),1,_max_ions_number))
            #       y.append(padding_data(np.zeros((2,2)),0,_max_ions_number))

            pred_ = session.run(pred_y,feed_dict={
                 batch_size:len(X),
                inputs_X: np.array(X),
                keep_prob:_keep_prob,
                seq_length:_seq_length[i],
                max_time:_max_ions_number
               
                                      })
            
            pred_[pred_>1]=1
            pred_[pred_<0]=0 
            _mse=session.run(MSE(tf.reshape(y,[-1,2]),pred_))
            mse_list.append(_mse)
            for k in range(len(X)): 
                for pred_s in pred_[k*_max_ions_number:k*_max_ions_number+_seq_length[i][k]]:
                    pred.append(pred_s)
        print("test_mse:"+str(np.mean(np.array(mse_list))))
        
    return test_idx,peptide,pred,merge_test_list

def get_merge_pred(merge_list,pred):
    print('get predict spectrum intensity list...')
    merge_list.append(datam.merge_list_2label(pred))
    return merge_list
def calc_pear(test_idx,peptide,pred,merge_list):
    
    pred_pd=pear.write_pred(test_idx,peptide,pred)
    merge_list=get_merge_pred(merge_list,pred_pd)
   
    person_mean=pear.get_pearson(merge_list) 
    return person_mean

def test(_batch_size,_keep_prob):
    test_idx,peptide,pred,merge_test_list=model_predict(_batch_size,_keep_prob)
    person_mean=calc_pear(test_idx,peptide,pred,merge_test_list)
    print(person_mean)

def main(args):
    
    if args.is_train==1:
        model_train(args)
    elif args.is_train==2:
        test(args.batch_size,args.keep_prob)
    else:
        train(args)
        test(args.batch_size,args.keep_prob)

if __name__ == '__main__':
   
   #with tf.device('/gpu:0'):
       #print('111')
   main(parse_args())
    