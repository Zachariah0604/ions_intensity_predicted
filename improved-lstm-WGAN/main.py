import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dirs = os.path.join( os.path.dirname(__file__),'..')
os.sys.path.append(os.path.join( os.path.dirname(__file__), '..'))
from get_data import *
from pearson import *
from get_batch import *
from model import *

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-train', type=int, default=2,help='1=train, 2=test, 0=train and test')
    parser.add_argument('--num-steps', type=int, default=2000,help='the number of training steps to take')
    parser.add_argument('--_lambda', type=float, default=10,help='Smaller lambda seems to help for toy tasks specifically')
    parser.add_argument('--mse-proportion', type=float, default=0.6,help='')
    parser.add_argument('--linear-initialization', type=str, default='None',help='')
    parser.add_argument('--batch-size', type=int, default=10,help='the number of peptide')
    parser.add_argument('--keep-prob', type=float, default=0.6,help='')
    parser.add_argument('--log-every', type=int, default=10,help='print loss after this many steps')
    
    parser.add_argument('--gan-input-size-G', type=int, default=281,help='')
    parser.add_argument('--gan-input-size-D', type=int, default=1,help='true ionic strength of ion')
    parser.add_argument('--D-hidden-size', type=int, default=20,help='')
    parser.add_argument('--learning-rate', type=float, default=1e-4,help='')

    parser.add_argument('--lstm-time-steps', type=int, default=1,help='')
    parser.add_argument('--lstm-input-size', type=int, default=281,help='')
    parser.add_argument('--lstm-output-size', type=int, default=1,help='predict ionic strength')
    parser.add_argument('--lstm-layer-num', type=int, default=2,help='')
    parser.add_argument('--lstm-cell-size', type=int, default=550,help='')
    return parser.parse_args()

def get_batch_peptide(merge_list,_batch_size):
    number_of_peptide=len(merge_list[0])
    batch_peptide=[]
    _batch_number=int(number_of_peptide/_batch_size)
    for i in range(_batch_number):
        batch_peptide.append(merge_list[0][i*_batch_size:(i+1)*_batch_size])
    if _batch_number*_batch_size < number_of_peptide:
        batch_peptide.append(merge_list[0][_batch_number*_batch_size:])
    return batch_peptide,_batch_number


def train(args):
    model = WGRAN(args._lambda,args.mse_proportion,args.linear_initialization, args.gan_input_size_G,args.gan_input_size_D,args.D_hidden_size,args.learning_rate,args.lstm_time_steps,
                 args.lstm_input_size,args.lstm_output_size,args.lstm_layer_num,args.lstm_cell_size)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("improved-lstm-WGAN-logs", sess.graph)
        sess.run(tf.global_variables_initializer())
        train_y,train_X,merge_train_list=get_train_data('data/data_swedcad_mm/am/train_seq_che.csv')

        


        batch_peptide,_batch_number=get_batch_peptide(merge_train_list,args.batch_size)
        print('..trainning')
        for Iter in range(args.num_steps):
            for i,(train_piptide_index) in enumerate(batch_peptide):
                train_ion_index=get_split_list(train_piptide_index)

                feed_dict= {model.real_data:train_y[np.array(train_ion_index)][:,np.newaxis],
                            model.lstm_X:train_X[np.array(train_ion_index)][:,np.newaxis,:],
                            model.keep_prob:args.keep_prob,
                            model.batch_size:len(train_ion_index)}
                if Iter > 0:
                        loss_g, _= sess.run([model.loss_g, model.gen_train_op], feed_dict=feed_dict)
                
                loss_d,_= sess.run(
                        [model.loss_d, model.disc_train_op],feed_dict=feed_dict)
            if Iter>0:   
                print('Iter:{0}/{1}  d-loss:{2:.4} --- g-loss:{3:.4}'.format(Iter,args.num_steps,loss_d,loss_g))
            else:
                print('Iter:{0}/{1}  d-loss:{2:.4}'.format(Iter,args.num_steps,loss_d))
            result = sess.run(merged, feed_dict=feed_dict)
            writer.add_summary(result, Iter)              
        print("SaveModel:",tf.train.Saver().save(sess,'improved-lstm-WGAN/model/gran-model.ckpt'))
           
     
  
def model_predict(_batch_size,_keep_prob):
    print('predicting..')
    test_idx,peptide,test_data,merge_test_list,label=get_test_data('data/data_swedcad_mm/am/test_seq_che.csv')
    print('number of peptide:'+str(len(merge_test_list[0])))
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        batch_peptide,_batch_number=get_batch_peptide(merge_test_list,_batch_size)

        saver = tf.train.import_meta_graph('improved-lstm-WGAN/model/gran-model.ckpt.meta')
        saver.restore(session, tf.train.latest_checkpoint('improved-lstm-WGAN/model/'))
        graph=tf.get_default_graph()
        X=graph.get_operation_by_name("Generator/lstm_X").outputs[0]
        real_data=graph.get_operation_by_name("Discriminator/D_x").outputs[0]
        batch_size=graph.get_operation_by_name("batch_size").outputs[0]
        keep_prob=graph.get_operation_by_name("keep_prob").outputs[0]
        y=tf.get_collection("pred_network")[0]
        loss_d=tf.get_collection("loss_d")[0]
        loss_g=tf.get_collection("loss_g")[0]
        disc_train_op=tf.get_collection("disc_train_op")[0]
        gen_train_op=tf.get_collection("gen_train_op")[0]

        pred=[]
        for i,(test_piptide_index) in enumerate(batch_peptide):
            test_ion_index=get_split_list(test_piptide_index)
            feed_dict={real_data:label[np.array(test_ion_index)][:,np.newaxis], 
                       X:test_data.values[test_ion_index][:,np.newaxis,:],
                       keep_prob:_keep_prob,
                       batch_size:len(test_ion_index),}
            test_loss_d,_= session.run([loss_d,disc_train_op],feed_dict=feed_dict)
            pred_,test_loss_g,_ = session.run([y,loss_g,gen_train_op],feed_dict=feed_dict)
            
            for pred_s in pred_:
                pred.append(pred_s[0])
            print("test_d-loss:"+str(test_loss_d)+" --- test_g-loss:"+str(test_loss_g))
    return test_idx,peptide,pred,merge_test_list

def get_merge_pred(merge_list,pred):
    print('get predict spectrum intensity list...')
    merge_list.append(get_merge_list(pred))
    return merge_list
def calc_pear(test_idx,peptide,pred,merge_list):
    pred_pd=write_pred(test_idx,peptide,pred)
    merge_list=get_merge_pred(merge_list,pred_pd)
    person_mean=get_pearson(merge_list) 
    return person_mean    
def test(_batch_size,_keep_prob):
    test_idx,peptide,pred,merge_test_list=model_predict(_batch_size,_keep_prob)
    person_mean=calc_pear(test_idx,peptide,pred,merge_test_list)
    print(person_mean)
def main(args):
    if(args.is_train==1):
        train(args)
    elif(args.is_train==2):
        test(args.batch_size,args.keep_prob)
    else:
        train(args)
        test(args.batch_size,args.keep_prob)
if __name__ == '__main__':
    main(parse_args())


