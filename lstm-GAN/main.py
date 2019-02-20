import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os
from scipy.stats import norm
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
    parser.add_argument('--is-train', type=int, default=0,help='1=train, 2=test, 0=train and test')
    parser.add_argument('--pre-num-steps', type=int, default=100,help='the number of pre training D network steps to take')
    parser.add_argument('--num-steps', type=int, default=10,help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=1,help='the number of peptide')
    parser.add_argument('--log-every', type=int, default=10,help='print loss after this many steps')

    parser.add_argument('--gan-input-size-G', type=int, default=281,help='')
    parser.add_argument('--gan-input-size-D', type=int, default=1,help='true ionic strength of ion')
    parser.add_argument('--gan-hidden-size', type=int, default=10,help='')
    parser.add_argument('--gan-learning-rate', type=float, default=0.001,help='')

    parser.add_argument('--lstm-time-steps', type=int, default=1,help='')
    parser.add_argument('--lstm-input-size', type=int, default=281,help='')
    parser.add_argument('--lstm-output-size', type=int, default=1,help='predict ionic strength')
    parser.add_argument('--lstm-layer-num', type=int, default=2,help='')
    parser.add_argument('--lstm-cell-size', type=int, default=300,help='')
    parser.add_argument('--lstm-learning-rate', type=float, default=0.00001,help='')
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
    model = GRAN(args.gan_input_size_G,args.gan_input_size_D,args.gan_hidden_size,args.gan_learning_rate,args.lstm_time_steps,
                 args.lstm_input_size,args.lstm_output_size,args.lstm_layer_num,args.lstm_cell_size,args.lstm_learning_rate)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("lstm-gan-logs", sess.graph)
        sess.run(tf.global_variables_initializer())
        train_y,train_X,merge_train_list=get_train_data('data/data_swedcad_mm/SwedCAD_all_ion2_filtered_by_resolution_for_debug_train.csv')
        batch_peptide,_batch_number=get_batch_peptide(merge_train_list,args.batch_size)

        print('pretraining discriminator')
        for Iter in range(args.pre_num_steps): 
            for i,(train_piptide_index) in enumerate(batch_peptide):
                train_ion_index=get_split_list(train_piptide_index)
                d = (np.random.random(len(train_ion_index)) - 0.5) * 10.0
                labels = train_y[np.array(train_ion_index)]
                pretrain_loss, _,d_pre = sess.run([model.pre_loss, model.pre_opt,model.D_pre], {
                    model.pre_input: np.reshape(d, (len(train_ion_index), 1)),
                    model.pre_labels: np.reshape(labels, (len(train_ion_index), 1)),
                    model.keep_prob:1.0,
                    model.batch_size:len(train_ion_index)
                })
                #print('{}--{}'.format(pretrain_loss,d_pre))
        weightsD = sess.run(model.d_pre_params)
        # copy weights from pre-training over to new D network
        for i, v in enumerate(model.d_params):
            sess.run(v.assign(weightsD[i]))

        print('..trainning')
        for Iter in range(args.num_steps):
            for i,(train_piptide_index) in enumerate(batch_peptide):
                train_ion_index=get_split_list(train_piptide_index)
                if i == 0:
                    feed_dict = {
                        model.D_x:train_y[np.array(train_ion_index)][:,np.newaxis],
                            model.lstm_X:train_X[np.array(train_ion_index)][:,np.newaxis,:],
                            model.lstm_y:train_y[np.array(train_ion_index)][:,np.newaxis,np.newaxis],
                            model.keep_prob:1.0,
                            model.batch_size:len(train_ion_index)
                    }
                else:
                    feed_dict = {
                        model.D_x:train_y[np.array(train_ion_index)][:,np.newaxis],
                        model.lstm_X:train_X[np.array(train_ion_index)][:,np.newaxis,:],
                        model.lstm_y:train_y[np.array(train_ion_index)][:,np.newaxis,np.newaxis],
                        model.batch_size:len(train_ion_index),
                        model.keep_prob:1.0,
                        model.lstm_cell_init_state: state  
                    }
                ''' update discriminator'''
                loss_d,_,d1,d2= sess.run(
                    [model.loss_d, model.opt_d,model.D1,model.D2],feed_dict={
                            model.D_x:train_y[np.array(train_ion_index)][:,np.newaxis],
                            model.lstm_X:train_X[np.array(train_ion_index)][:,np.newaxis,:],
                            model.lstm_y:train_y[np.array(train_ion_index)][:,np.newaxis,np.newaxis],
                            model.keep_prob:1.0,
                            model.batch_size:len(train_ion_index)
                        })

                ''' update generator '''
                loss_g,_, state, pred= sess.run([model.loss_g, model.opt_g, model.lstm_cell_final_state, model.lstm_pred], feed_dict=feed_dict)


                #if Iter % args.log_every == 0:
                #    print('{0}:batch-step:{1}/{2} data-size:{3} loss-lstm:{4:.4}'.format(Iter+args.log_every,i+1,_batch_number+1, len(train_ion_index),lstm_loss))
                #    print('Iter:{0}/{1}  loss-d:{2:.4} loss-g:{3:.4} '.format(Iter+args.log_every,args.num_steps,loss_d, loss_g))       
            print('Iter:{0}/{1}  loss-d:{2:.4} loss-g:{3:.4}--- '.format(Iter,args.num_steps,loss_d, loss_g))   
            #result = sess.run(merged, feed_dict=feed_dict)
            #writer.add_summary(result, Iter)
            #if Iter % args.log_every == 0:
                #print('Iter:{0}/{1}  loss-d:{2:.4} loss-g:{3:.4} '.format(Iter+args.log_every,args.num_steps,loss_d, loss_g))                
        print("SaveModel:",tf.train.Saver().save(sess,'lstm-GAN/model/gran-model.ckpt'))
           
def MSE(label,pred):
    return tf.reduce_mean(tf.square(np.array(pred[:,0])-np.array(label[:,0][:,0])))      
  
def model_predict(_batch_size):
    print('predicting..')
    test_idx,peptide,test_data,merge_test_list,label=get_test_data('data/data_swedcad_mm/test_SwedCAD_spectra.csv')
    print('number of peptide:'+str(len(merge_test_list[0])))
    with tf.Session() as session:
        
        session.run(tf.global_variables_initializer())
        batch_peptide,_batch_number=get_batch_peptide(merge_test_list,_batch_size)
        saver = tf.train.import_meta_graph('lstm-GAN/model/gran-model.ckpt.meta')
        saver.restore(session, tf.train.latest_checkpoint('lstm-GAN/model/'))
        graph=tf.get_default_graph()
        X=graph.get_operation_by_name("Generator/lstm_inputs/lstm_X").outputs[0]
        batch_size=graph.get_operation_by_name("batch_size").outputs[0]
        keep_prob=graph.get_operation_by_name("keep_prob").outputs[0]
        y=tf.get_collection("pred_network")[0]
  
        pred=[]
        for i,(test_piptide_index) in enumerate(batch_peptide):
            test_ion_index=get_split_list(test_piptide_index)
            pred_ = session.run(
                y,feed_dict={
                            X:test_data.values[test_ion_index][:,np.newaxis,:],
                            keep_prob:1.0,
                            batch_size:len(test_ion_index),
                },
              )
            _mse=session.run(MSE(label[test_ion_index][:,np.newaxis,np.newaxis],pred_))
            for pred_s in pred_:
                pred.append(pred_s[0])
            print("test_mse:"+str(_mse))
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
def test(_batch_size):
    test_idx,peptide,pred,merge_test_list=model_predict(_batch_size)
    person_mean=calc_pear(test_idx,peptide,pred,merge_test_list)
    print(person_mean)
def main(args):
    if(args.is_train==1):
        train(args)
    elif(args.is_train==2):
        test(args.batch_size)
    else:
        train(args)
        test(args.batch_size)
if __name__ == '__main__':
    main(parse_args())


