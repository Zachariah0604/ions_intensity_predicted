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

class svm3calss(object):
    def __init__(self,  args):
        self.input_size = args.input_size
        self.num_classes =  args.num_classes
        self.learning_rate=args.learning_rate
        
  

        self.inputs = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.targets = tf.placeholder(shape=[self.num_classes, None], dtype=tf.float32) 
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')

        self.bias = tf.Variable(tf.random_normal(shape=[self.num_classes,self.batch_size]))
       
        with tf.name_scope('rbf_kernel'):
            gamma=self.rbf_kernel()
        with tf.variable_scope('loss'):
            self.log_loss()
        with tf.variable_scope('predict_kernel'):
            self.rbf_predict_kernel(gamma)
        with tf.variable_scope('predict'):
            self.predict()
        with tf.name_scope('train'):
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def rbf_kernel(self):
        gamma = tf.constant(-10.0)
        dist = tf.reduce_sum(tf.square(self.inputs), 1)
        dist = tf.reshape(dist, [-1,1])
        sq_dists = tf.multiply(2., tf.matmul(self.inputs, tf.transpose(self.inputs)))
        self.kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
        return gamma
    def reshape_matmul(self,mat):
        v1 = tf.expand_dims(mat, 1)
        v2 = tf.reshape(v1, [self.num_classes, self.batch_size, 1])
        return(tf.matmul(v2, v1))

    def log_loss(self):
        first_term = tf.reduce_sum(self.bias)
        b_vec_cross = tf.matmul(tf.transpose(self.bias), self.bias)
        y_target_cross = self.reshape_matmul(self.targets)
        second_term = tf.reduce_sum(tf.multiply(self.kernel , tf.multiply(b_vec_cross, y_target_cross)),[1,2])
        self.loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))
        
       
    def rbf_predict_kernel(self,gamma):
        self.prediction_grid = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        rA = tf.reshape(tf.reduce_sum(tf.square(self.inputs), 1),[-1,1])
        rB = tf.reshape(tf.reduce_sum(tf.square(self.prediction_grid), 1),[-1,1])
        pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(self.inputs, tf.transpose(self.prediction_grid)))), tf.transpose(rB))
        self.pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
    def predict(self):
        prediction_output = tf.matmul(tf.multiply(self.targets,self.bias), self.pred_kernel) 
        self.prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0) 
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.targets,0)), tf.float32))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-train', type=int, default=0,help='1=train, 2=test, 0=train and test')
    parser.add_argument('--num-iter', type=int, default=100,help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=64,help='the number of peptide')

    parser.add_argument('--learning-rate', type=float, default=1e-3,help='')
    parser.add_argument('--input-size', type=int, default=192,help='')
    parser.add_argument('--num-classes', type=int, default=3,help='')


    parser.add_argument('--intensity_num_label', type=int, default=4, help="") 
    return parser.parse_args()


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



def model_train(args):
    model = svm3calss(args)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("seq2seq-WGAN/svm-logs", sess.graph)
        sess.run(tf.global_variables_initializer())
         #train data
        _,_,train_X,train_y,merge_train_list=data.get_discretization_data('data/data_mm/test_test.txt',args.num_classes)
        print(str(len(merge_train_list[0]))+' train peptides ,DataShape:('+str(np.array(train_X).shape)+str(np.array(train_y).shape)+')')
        train_batch_peptide,_batch_number,seq_length=get_batch_peptide(merge_train_list,args.batch_size)
        y1 = np.array([1 if y==0 else -1 for y in train_y]) 
        y2 = np.array([1 if y==1 else -1 for y in train_y]) 
        y3 = np.array([1 if y==2 else -1 for y in train_y]) 
        train_y = np.array([y1, y2, y3])

        ##val data
        #_,_,test_X,test_y,merge_test_list=data.get_discretization_data('data/data_mm/test.txt',args.num_classes)
        #print(str(len(merge_test_list[0]))+' test peptides ,DataShape:('+str(np.array(test_X).shape)+str(np.array(test_y).shape)+')')
    
      
        print('..trainning')
        loss_vec = []
        batch_accuracy = []
        for Iter in range(args.num_iter):
            for i,train_piptides in enumerate(train_batch_peptide):

                train_ions=data.get_split_list2(train_piptides)
                
                X= train_X[train_ions]
                y= train_y[:,train_ions]
                sess.run(model.train_op, feed_dict={model.inputs: X, model.targets:  y,model.batch_size:len(X)})
                temp_loss = sess.run(model.loss, feed_dict={model.inputs:  y, model.targets:  y,model.batch_size:len(X)})
                loss_vec.append(temp_loss) 

                acc_temp = sess.run(model.accuracy, feed_dict={model.inputs: X, model.targets:  y, model.prediction_grid:X,model.batch_size:len(X)})
                batch_accuracy.append(acc_temp) 

                print('Step #' + str(i+1))
                print('Loss = ' + str(temp_loss))


            
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
    