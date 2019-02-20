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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-train', type=int, default=0,help='1=train, 2=test, 0=train and test')
    parser.add_argument('--num-iter', type=int, default=10,help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=64,help='the number of peptide')
    parser.add_argument('--keep-prob', type=float, default=0.5,help='')
    parser.add_argument('--learning-rate', type=float, default=1e-3,help='')
    parser.add_argument('--input-size', type=int, default=88,help='')
    parser.add_argument('--num_classes', type=int, default=4,help='predict ionic strength')
    parser.add_argument('--layer-num', type=int, default=2,help='')
    parser.add_argument('--cell-size', type=int, default=256,help='')
    return parser.parse_args()

class LSTM(object):
    def __init__(self,  args):
        self.max_time = tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.input_size = args.input_size
        self.num_classes =  args.num_classes
        self.cell_size =  args.cell_size
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')

        self.layer_num=args.layer_num
        self.learning_rate=args.learning_rate
        self.keep_prob=args.keep_prob
        self.seq_length = tf.placeholder(tf.float32,[None],name='seq_length')
       
       
        self.X = tf.placeholder(tf.float32, [None, None, self.input_size], name='X')
        self.y = tf.placeholder(tf.float32, [None, None, self.num_classes], name='y')
        print(self.y.get_shape())
        self.create_net()
     
           

    def create_net(self,):
        l_in_x = tf.reshape(self.X, [-1, self.input_size])
        Ws_in = tf.get_variable(name='ws_in',shape=[self.input_size, self.cell_size])
        bs_in = tf.get_variable(name='bs_in',shape=[self.cell_size,])
        cell_inputs = tf.reshape(tf.nn.relu(tf.matmul(l_in_x, Ws_in) + bs_in), [self.batch_size, self.max_time, self.cell_size])


        lstm_fw_cell =tf.nn.rnn_cell.LSTMCell(self.cell_size)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.cell_size)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=self.keep_prob)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=self.keep_prob)
        mlstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * self.layer_num, state_is_tuple=True)
        mlstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * self.layer_num, state_is_tuple=True)
        self.seq_length = tf.cast(self.seq_length, tf.int32) 
        (self.output_fw, self.output_bw), self.states = tf.nn.bidirectional_dynamic_rnn(
                                                                mlstm_fw_cell,
                                                                mlstm_bw_cell,
                                                                cell_inputs,
                                                                sequence_length=self.seq_length,
                                                                dtype=tf.float32 )

        l_out_x = tf.reshape(tf.concat([self.output_fw, self.output_bw],axis=2), [-1, self.cell_size * 2])
        Ws_out = tf.get_variable(name='ws_out',shape=[self.cell_size*2, self.num_classes])
        bs_out = tf.get_variable(name='bs_out',shape=[self.num_classes, ])
        outputs =tf.nn.sigmoid(tf.matmul(l_out_x, Ws_out) + bs_out)
      

        self.pred_tags = tf.reshape(outputs, [-1, self.max_time, self.num_classes])
        print(self.pred_tags.get_shape())
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_tags,
                                                                    labels=self.y)
        mask = tf.sequence_mask(tf.cast(self.seq_length, tf.int32))
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)
        
        
        tf.add_to_collection('pred_tags', self.tags)
        tf.summary.scalar('loss', self.loss)
            
        self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,var_list=self._params)
   
    @staticmethod
    

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

def padding_data(data,max_ions_number):
    _ydim=data.shape[1]
    dv=max_ions_number-data.shape[0]
    data=data.tolist()
    if dv > 0:
        data.extend(np.zeros((dv,_ydim)).tolist())
    return data

def train(args):
    model = LSTM(args)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("lstm/lstm-logs", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        
        #train data
        _,_,train_X,train_y,merge_train_list=data.get_discretization_data('data/data_mm/train.txt',args.num_classes)
        print(str(len(merge_train_list[0]))+' train peptides ,DataShape:('+str(np.array(train_X).shape)+str(np.array(train_y).shape)+')')
        train_batch_peptide,train_batch_number,train_seq_length=get_batch_peptide(merge_train_list,args.batch_size)

        #val data
        _,_,val_X,val_y,merge_val_list=data.get_discretization_data('data/data_mm/test.txt',args.num_classes)
        print(str(len(merge_val_list[0]))+' val peptides ,DataShape:('+str(np.array(val_X).shape)+str(np.array(val_y).shape)+')')
        val_batch_peptide,val_batch_number,val_seq_length=get_batch_peptide(merge_val_list,args.batch_size)
        
        batch_peptide,_batch_number,seq_length=get_batch_peptide(merge_train_list,args.batch_size)
       
        print('..trainning')

        for Iter in range(args.num_iter):
            
            #train
            train_loss=0.0;train_accuracy=0.0
            for i,train_piptides_index in enumerate(train_batch_peptide):
                X=[];y=[];
                max_ions_number=max(train_seq_length[i])
                for j in range(len(train_piptides_index)):
                    train_ions_index=datam.get_split_list(train_piptides_index[j])
                    X.append(padding_data(train_X[np.array(train_ion_index)],max_ions_number))
                    y.append(padding_data(train_y[np.array(train_ion_index)],max_ions_number))
               
                feed_dict = {
                        model.X:np.array(X,dtype=np.float32),
                        model.y:np.array(y,dtype=np.float32),
                        model.seq_length:train_seq_length[i],
                        model.max_time:max_ions_number,
                        model.batch_size:len(X)
                        
                }
                _, loss, state, pred_tags = sess.run(
                    [model.train_op, model.loss, model.pred_tags],feed_dict=feed_dict)
                train_loss+=loss
                pred_tags[pred_tags>0.5]=1
                pred_tags[pred_tags<=0.5]=0
                mask = (np.expand_dims(np.arange(max_ions_number), axis=0) < np.expand_dims(train_seq_length[i], axis=1))
                total_labels = np.sum(train_seq_length[i])
                correct_labels = np.sum((np.array(y) == pred_tags) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                train_accuracy+=accuracy 
            #val
            val_loss=0.0;val_accuracy=0.0
            for i, val_piptide_index in enumerate(val_batch_peptide):
                X=[];y=[]
               
                max_ions_number=max(val_seq_length[i])
                for j in range(len(val_piptide_index)):
                    train_ion_index=data.get_split_list(val_piptide_index[j])
                    X.append(padding_data(val_X[np.array(train_ion_index)],max_ions_number))
                    y.append(padding_data(val_y[np.array(train_ion_index)],max_ions_number))
                feed_dict_val = {
                            model.X:np.array(X),
                            model.y:np.array(y),
                            model.seq_length:val_seq_length[i],
                            model.max_time:max_ions_number,
                            model.batch_size:len(X)
                            
                    }
                loss, pred_tags = sess.run([model.loss, model.pred_tags],feed_dict=feed_dict_val)
                val_loss+=loss
                mask = (np.expand_dims(np.arange(max_ions_number), axis=0) < np.expand_dims(val_seq_length[i], axis=1))
                total_labels = np.sum(val_seq_length[i])
                correct_labels = np.sum((np.array(y) == pred_tags) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                val_accuracy+=accuracy 

            result = sess.run(merged, feed_dict)
            writer.add_summary(result, Iter)
            
            print("Epoch: %d" % (Iter+1), "train loss: %.2f" % (train_loss/_batch_number),"train acc: %.2f%%" % (train_acc/train_batch_number))
            print("Epoch: %d" % (Iter+1), "val loss: %.2f" % (val_loss/val_batch_number),"val acc: %.2f%%" % (val_acc/val_batch_number))
        print("SaveModel:",tf.train.Saver().save(sess,'lstm/model2/bilstm_model.ckpt'))
 
        

  
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
 
   datam=GetData(4)
   main(parse_args())
    