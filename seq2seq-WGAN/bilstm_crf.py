import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import random



import sys 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dirs = os.path.join( os.path.dirname(__file__),'..')
os.sys.path.append(os.path.join( os.path.dirname(__file__), '..'))
from tools.get_data import GetData
from tools.pearson import CalcPerson

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-train', type=int, default=0,help='1=train, 2=test, 0=train and test')
    parser.add_argument('--num-iter', type=int, default=100,help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=128,help='the number of peptide')
    parser.add_argument('--keep-prob', type=float, default=0.5,help='')
    parser.add_argument('--learning-rate', type=float, default=1e-3,help='')
    parser.add_argument('--input-size', type=int, default=88,help='')
    parser.add_argument('--num-classes', type=int, default=2,help='')
    parser.add_argument('--output-size', type=int, default=1,help='predict ionic strength')
    parser.add_argument('--layer-num', type=int, default=2,help='')
    parser.add_argument('--cell-size', type=int, default=650,help='')
    parser.add_argument('--intensity_num_label', type=int, default=4, help="") 
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
        self.num_classes=args.num_classes
        self.keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')
        self.seq_length = tf.placeholder(tf.float32,[None],name='seq_length')
        #self.batch_size = batch_size
       
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [None, None, self.input_size], name='X')
            self.y = tf.placeholder(tf.int32, [None, None], name='y')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.variable_scope('loss'):
            self.add_crf_layer()
        with tf.name_scope('train'):
            self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            #regularization= 0.001* tf.reduce_sum([ tf.nn.l2_loss(v) for v in self._params ])
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,var_list=self._params)
    def add_input_layer(self,):
        l_in_x = tf.reshape(self.X, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        
        bs_in = self._bias_variable([self.cell_size,])
        with tf.name_scope('Wx_plus_b'):
            l_in_y =tf.matmul(l_in_x, Ws_in) + bs_in
        self.l_in_y = tf.reshape(l_in_y, [-1, self.max_time, self.cell_size], name='2_3D')

    def add_cell(self):
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
        
        l_out_x =tf.reshape(tf.concat([self.output_fw, self.output_bw],axis=2), [-1, self.cell_size * 2])
        Ws_out = self._weight_variable([self.cell_size*2, self.num_classes])
        tf.summary.histogram('Ws_out',Ws_out)
        bs_out = self._bias_variable([self.num_classes, ])
        tf.summary.histogram('bs_out',bs_out)
        with tf.name_scope('Wx_plus_b'):
            self.lstm_outputs =tf.matmul(l_out_x, Ws_out) + bs_out
       
        

    def add_crf_layer(self):
        scores = tf.reshape(self.lstm_outputs, [-1, self.max_time, self.num_classes])

        
        
        if True:
            # Linear-CRF.
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(scores, tf.reshape(self.y,[-1,self.max_time]),tf.cast(self.seq_length, tf.int32)) #loss=MLP(pred,lable)

            self.loss = tf.reduce_mean(-log_likelihood)

            self.tags, best_score = tf.contrib.crf.crf_decode(scores, self.transition_params, tf.cast(self.seq_length, tf.int32))

            #reshape_label=tf.reshape(tf.cast(self.y,tf.float32),[-1,1])
            #reshape_tags=tf.reshape(tf.cast(self.tags,tf.float32),[-1,1])


            #self.mse_bias=tf.losses.mean_squared_error(reshape_tags,reshape_label)

            #self.loss+=(self.mse_bias*0.1)

            #pred=1/(1+tf.exp(-tf.reshape(tf.cast(self.tags,tf.float32),[-1,1])))
            #lable_p=1/(1+tf.exp(-tf.reshape(tf.cast(self.y,tf.float32),[-1,1])))
            #cross_entropy = -lable_p * tf.log(pred) -(1-lable_p) * tf.log(1-pred)
            #reduce_sum = tf.reduce_sum(cross_entropy, 1)
            #rank_loss = tf.reduce_mean(reduce_sum)
            #self.loss+=rank_loss

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores,
                                                                    labels=self.y)
            mask = tf.sequence_mask(tf.cast(self.seq_length, tf.int32))
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

            self.tags = tf.argmax(scores, axis=-1)
            self.tags = tf.cast(self.tags, tf.int32)
        tf.summary.scalar('loss', self.loss)
        tf.add_to_collection('pred_network', self.tags)
        


    #def compute_cost(self,pred,label):
    #    losses = tf.contrib.nn.seq2seq.sequence_loss_by_example(
    #        [tf.reshape(pred, [-1], name='reshape_pred')],
    #        [tf.reshape(label, [-1], name='reshape_target')],
    #        [tf.ones([self.batch_size * self.max_time], dtype=tf.float32)],
    #        average_across_timesteps=True,
    #        softmax_loss_function=self.ms_error,
    #        name='losses'
    #    )
    #    loss = tf.div(
    #            tf.reduce_sum(losses, name='losses_sum'),
    #            tf.cast(self.batch_size,tf.float32),
    #            name='average_loss')
    #    return loss
            #tf.summary.scalar('loss', self.loss)
    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        #initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape,  name=name)

    def _bias_variable(self, shape, name='biases'):
        #initializer = tf.constant_initializer(0.1)
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
        for k in range(len(batch_peptide[-1])):
            seq_length[len(batch_peptide)-1].append(len(merge_list[0][_batch_number*_batch_size+k]))
        _batch_number+=1
    return batch_peptide,_batch_number,seq_length

def padding_data(data,flag,max_ions_number):
    if flag ==1 :
        _ydim=data.shape[1]
    else:
        _ydim=data.shape[0]
    #_ydim=data.shape[1]
    dv=max_ions_number-data.shape[0]
    data=data.tolist()
    if dv > 0:
        if flag ==1:
            data.extend(np.zeros((dv,_ydim)).astype('int32').tolist())
        else:
            data.extend(np.zeros((dv,)).astype('int32').tolist())
        #data.extend(np.zeros((dv,_ydim)).tolist())
    return data

def model_train(args):
    model = LSTM(args)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("lstm-logs", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        #train data
        _,_,train_X,train_y,merge_train_list=data.get_discretization_data('data/data_mm/train.txt',args.num_classes)
        print(str(len(merge_train_list[0]))+' train peptides ,DataShape:('+str(np.array(train_X).shape)+str(np.array(train_y).shape)+')')
        batch_peptide,_batch_number,seq_length=get_batch_peptide(merge_train_list,args.batch_size)

        #val data
        _,_,val_X,val_y,merge_val_list=data.get_discretization_data('data/data_mm/test.txt',args.num_classes)
        print(str(len(merge_val_list[0]))+' val peptides ,DataShape:('+str(np.array(val_X).shape)+str(np.array(val_y).shape)+')')
        val_batch_peptide,val_batch_number,val_seq_length=get_batch_peptide(merge_val_list,args.batch_size)
    
        #if len(seq_length[-1]) < args.batch_size:
        #   seq_length[-1].extend([0]*(args.batch_size-len(seq_length[-1])))
        print('..trainning')
       
        for Iter in range(args.num_iter):
            train_acc=0;train_loss=0
            permutation_batch = np.random.permutation(len(batch_peptide))
            suffled_batch_peptide=np.array(batch_peptide)[permutation_batch].tolist()
            suffled_seq_length=np.array(seq_length)[permutation_batch].tolist()
            for i,(train_piptide_index) in enumerate(suffled_batch_peptide):
                X=[];y=[];
                max_ions_number=max(suffled_seq_length[i])
                permutation_peptide = np.random.permutation(len(train_piptide_index))
                suffled_seq=np.array(suffled_seq_length[i])[permutation_peptide].tolist()
                suffled_train_piptide_index=np.array(train_piptide_index)[permutation_peptide].tolist()
                #padding_pep_num=args.batch_size-len(train_piptide_index)
                for j in range(len(suffled_train_piptide_index)):
                    train_ion_index=data.get_split_list(suffled_train_piptide_index[j])
                    X.append(padding_data(train_X[np.array(train_ion_index)],1,max_ions_number))
                    y.append(padding_data(train_y[np.array(train_ion_index)],1,max_ions_number))
                #if padding_pep_num >0:
                #   for k in range(padding_pep_num):
                       
                #       X.append(padding_data(np.zeros((2,args.input_size)),1,max_ions_number))
                #       y.append(padding_data(np.zeros((2,)),0,max_ions_number))
                
                feed_dict = {
                        model.X:np.array(X),
                        model.y:np.array(y),
                        model.keep_prob:args.keep_prob,
                        model.seq_length:suffled_seq,
                        model.max_time:max_ions_number,
                        model.batch_size:len(X)
                        
                }
                _, loss, state, pred = sess.run(
                    [model.train_op, model.loss,model.states, model.tags],
                    feed_dict=feed_dict)
                train_loss+=loss            
                mask = (np.expand_dims(np.arange(max_ions_number), axis=0) < np.expand_dims(suffled_seq, axis=1))
                total_labels = np.sum(suffled_seq)
                correct_labels = np.sum((np.array(y) == pred) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                train_acc+=accuracy
            val_acc=0;val_loss=0
            for i, val_piptide_index in enumerate(val_batch_peptide):
                X=[];y=[]
               
                max_ions_number=max(val_seq_length[i])
                for j in range(len(val_piptide_index)):
                    train_ion_index=data.get_split_list(val_piptide_index[j])
                    X.append(padding_data(val_X[np.array(train_ion_index)],1,max_ions_number))
                    y.append(padding_data(val_y[np.array(train_ion_index)],0,max_ions_number))
                feed_dict_val = {
                            model.X:np.array(X),
                            model.y:np.array(y),
                            model.keep_prob:args.keep_prob,
                            model.seq_length:val_seq_length[i],
                            model.max_time:max_ions_number,
                            model.batch_size:len(X)
                            
                    }
                loss_val, pred_val = sess.run([model.loss, model.tags],feed_dict=feed_dict_val)
                val_loss+=loss_val
                mask = (np.expand_dims(np.arange(max_ions_number), axis=0) < np.expand_dims(val_seq_length[i], axis=1))
                total_labels = np.sum(val_seq_length[i])
                correct_labels = np.sum((np.array(y) == pred_val) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                val_acc+=accuracy 
            #val_acc,val_loss=val(args,model,val_X,val_y,val_batch_peptide,val_seq_length)
            print("Epoch: %d" % (Iter+1), "train loss: %.2f" % (train_loss/_batch_number),"train acc: %.2f%%" % (train_acc/_batch_number))
            print("Epoch: %d" % (Iter+1), "val loss: %.2f" % (val_loss/val_batch_number),"val acc: %.2f%%" % (val_acc/val_batch_number))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, Iter)
            
            #print('Iter[%d/%d],loss[%.4f]' % (Iter+1,args.num_iter,round(loss,4)))
        print("SaveModel:",tf.train.Saver().save(sess,'lstm/model/model.ckpt'))
 
        
def MSE(label,pred):
    return tf.reduce_mean(tf.square(pred-label))   
  
def model_predict(args,kmodel,test_data,merge_test_list,test_label):
    print('predicting..')
    
    print('number of peptide:'+str(len(merge_test_list[0])))
    with tf.Session() as session:
        batch_peptide,_batch_number,_seq_length=get_batch_peptide(merge_test_list,args.batch_size)
        

        saver = tf.train.import_meta_graph('lstm/model/model.ckpt.meta')
        saver.restore(session, tf.train.latest_checkpoint('lstm/model/'))
        graph=tf.get_default_graph()
        inputs_X=graph.get_operation_by_name("inputs/X").outputs[0]
        batch_size=graph.get_operation_by_name("batch_size").outputs[0] 
        keep_prob=graph.get_operation_by_name("keep_prob").outputs[0]
        max_time=graph.get_operation_by_name("max_time").outputs[0]
        seq_length=graph.get_operation_by_name("seq_length").outputs[0]

        pred_y=tf.get_collection("pred_network")[0]
        pred=[];aaa=[]
        mse_list=[]
        total_labels=0;correct_labels=0
        for i,(test_piptide_index) in enumerate(batch_peptide):
            X=[];y=[]
            _max_ions_number=max(_seq_length[i])
            padding_pep_num=args.batch_size-len(test_piptide_index)
            for j in range(len(test_piptide_index)):
                test_ion_index=get_split_list(test_piptide_index[j])
                X.append(padding_data(test_data[np.array(test_ion_index)],1,_max_ions_number))
                y.append(padding_data(test_label[np.array(test_ion_index)],0,_max_ions_number))
            

            pred_ = session.run(pred_y,feed_dict={
                inputs_X: np.array(X),
                keep_prob:args.keep_prob,
                seq_length:_seq_length[i],
                max_time:_max_ions_number,
                batch_size:len(X)
                                      })
            for k in range(len(X)): 
                pred.extend(pred_[k][:_seq_length[i][k]])
                aaa.extend(y[k][:_seq_length[i][k]])

            mask = (np.expand_dims(np.arange(_max_ions_number), axis=0) < np.expand_dims(_seq_length[i], axis=1))
            total_labels += np.sum(_seq_length[i])
            correct_labels += np.sum((np.array(y) == pred_) * mask)
        
        accuracy = 100.0 * correct_labels / float(total_labels)
        print("test Accuracy: %.2f%%" % accuracy)
            #pred_[pred_>1]=1
            #pred_[pred_<0]=0 
            #_mse=session.run(MSE(np.reshape(y,(-1,1)),pred_))
            #mse_list.append(_mse)
        cunt=0;cunt2=0
        for i in range(len(aaa)):
            if aaa[i]==0:
                cunt+=1
            if aaa[i]==pred[i]:
                cunt2+=1
               
        print(cunt/len(aaa))
        print(cunt2/len(aaa))
        predaaa = pd.DataFrame({"pred":pred,"label":aaa})
        predaaa.to_csv('data//SwedCAD_pred2.csv')
        min_max_scaler = preprocessing.MinMaxScaler()
        pred_minmax = min_max_scaler.fit_transform(pred)
    return pred_minmax

def get_merge_pred(merge_list,pred,data):
    print('get predict spectrum intensity list...')
    merge_list.append(data.merge_list_1label(pred))
    return merge_list
def calc_pear(test_idx,peptide,pred,merge_list,pear,data):
    
    pred_pd=pear.write_pred(test_idx,peptide,pred)
    merge_list=get_merge_pred(merge_list,pred_pd,data)
   
    person_mean=pear.get_pearson(merge_list) 
    return person_mean

def test(args,data,pear):
    test_idx,peptide,pred,merge_test_list=model_predict(args,data)
    person_mean=calc_pear(test_idx,peptide,pred,merge_test_list,pear,data)
    print(person_mean)

def main(args,data,pear):
    if args.is_train==1:
        model_train(args)
    elif args.is_train==2:
        test(args,data,pear)
    else:
        model_train(args)
        test(args,data,pear)

if __name__ == '__main__':
    args=parse_args()
    
    data=GetData(args.intensity_num_label)
    pear=CalcPerson(args.intensity_num_label)
    main(args,data,pear)
    