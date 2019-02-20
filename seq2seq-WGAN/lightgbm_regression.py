import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import random
from sklearn.externals import joblib
import lightgbm as lgb
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score
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
    parser.add_argument('--num-classes', type=int, default=3,help='')
    parser.add_argument('--intensity_num_label', type=int, default=4, help="") 
    return parser.parse_args()




def get_params():
    params = {}
    #params["boosting"]="gbdt" 
    #params["boost_from_average"]="false"
    #params["application"]="multiclass"
    params["num_leaves"]=63
    params["min_data_in_leaf"]=100
    #params["max_depth"] =8
    params["n_estimators"]=2000
    params["learning_rate"]=0.05
    params["feature_fraction"]=0.8
    params["bagging_freq"]=50
    params["bagging_fraction"]=0.8
    params['verbose'] = 0
    params["is_unbalance"]="true"
 
    #params['device']='gpu'
  
    #params['num_class']=3
    return params


def model_train(args):
   
    #train data
    _,_,train_X,train_y,merge_train_list=data.get_data('data/data_mm/test_test.txt',True)
    print(str(len(merge_train_list[0]))+' train peptides ,DataShape:('+str(np.array(train_X).shape)+str(np.array(train_y).shape)+')')
    #batch_peptide,_batch_number,seq_length=get_batch_peptide(merge_train_list,args.batch_size)

    #val data
    _,_,test_X,test_y,merge_test_list=data.get_data('data/data_mm/test_test.txt',True)
    print(str(len(merge_test_list[0]))+' test peptides ,DataShape:('+str(np.array(test_X).shape)+str(np.array(test_y).shape)+')')
    #test_batch_peptide,test_batch_number,test_seq_length=get_batch_peptide(merge_test_list,args.batch_size)

    print('..trainning')
    best_loss=0.0
    
        
        
       # for i,train_piptide in enumerate(merge_train_list[0]):
            #print(str(i)+"/"+str(_batch_number))
    cv=KFold(len(merge_train_list[0]),10,shuffle=True,random_state=int(2018))
    for i,(cv_train_peptide,cv_val_peptide) in enumerate(cv):
           
        cv_train_ions=data.get_split_list2(np.array(merge_train_list[0])[cv_train_peptide]) 
        cv_val_ions=data.get_split_list2(np.array(merge_train_list[0])[cv_val_peptide]) 
            
        train_dataset = lgb.Dataset(train_X[np.array(cv_train_ions)],train_y[np.array(cv_train_ions)])
        eval_dataset=lgb.Dataset(train_X[np.array(cv_val_ions)],train_y[np.array(cv_val_ions)],reference=train_dataset)
        gbm=lgb.LGBMRegressor(n_estimators=3000,learning_rate=0.1,num_leaves=127)
        gbm.fit(train_X[np.array(cv_train_ions)],train_y[np.array(cv_train_ions)],eval_set=[(train_X[np.array(cv_val_ions)],train_y[np.array(cv_val_ions)])],early_stopping_rounds=50)
        #gbm = lgb.train(get_params(),train_dataset,num_boost_round=5000,valid_sets=eval_dataset,verbose_eval=250,early_stopping_rounds=50)
        print(gbm.best_iteration_)
        pred_label=gbm.predict(train_X[np.array(cv_val_ions)], num_iteration=gbm.best_iteration_)  
    

        total_labels = len(cv_val_ions)
        correct_labels = np.sum(train_y[np.array(cv_val_ions)] == pred_label)
        val_accuracy = round(100.0 * correct_labels / float(total_labels),3)
            
        
       #for i, val_piptide in enumerate(val_batch_peptide):

        test_ions=data.get_split_list2(merge_test_list[0])  
        test_pred_label=gbm.predict(test_X[np.array(test_ions)],num_iteration=gbm.best_iteration_)
        #test_pred_label=np.argmax(test_pred_prob, axis=1)

        total_labels = len(test_ions)
        correct_labels = np.sum(test_y[np.array(test_ions)] == test_pred_label)
        test_accuracy = round(100.0 * correct_labels / float(total_labels),3)
             
        
            
        print("cv"+str(i+1)+":"+str(len(cv_train_peptide))+" train peptides, "+str(len(cv_train_ions))+" train ions ")
        print("cv"+str(i+1)+":"+str(len(cv_val_peptide))+" val peptides, "+str(len(cv_val_ions))+" val ions,"+str(val_accuracy)+" val acc")
        print("cv"+str(i+1)+":"+str(len(merge_test_list[0]))+" test peptides, "+str(len(test_ions))+" test ions, "+str(test_accuracy)+" test acc")
        if test_accuracy>best_acc:
            print("best test accuracy is " +str(test_accuracy)+", saved best model gbm.pkl\n-------------------------------\n")
            best_acc=test_accuracy
            joblib.dump(gbm,'seq2seq-WGAN/model/gbm.pkl')

def model_predict(args):
    print('predicting..')
    
    peptide_idxs,peptides,test_X,test_y,merge_test_list=data.get_discretization_data('data/data_mm/test_test.txt',args.num_classes)
    print(str(len(merge_test_list[0]))+' test peptides ,DataShape:('+str(np.array(test_X).shape)+str(np.array(test_y).shape)+')')    
    
    gbm_model=joblib.load('seq2seq-WGAN/model/gbm.pkl')
    test_ions=data.get_split_list2(merge_test_list[0])  
    test_pred_label=gbm_model.predict(test_X[np.array(test_ions)])


    total_labels = len(test_ions)
    correct_labels = np.sum(test_y== test_pred_label)
    test_accuracy = round(100.0 * correct_labels / float(total_labels),3)
    print(test_accuracy)
    return peptide_idxs,peptides,test_pred_label




def test(args):
    peptide_idxs,peptides,test_pred_label=model_predict(args)
    with open('data/data_mm/gbm_classfication_test.txt','w') as f:
        pred_label = np.array(test_pred_label).reshape([-1,4])
        for i in range(len(pred_label)):
            peptide=peptides[i*4]
            i_label=','.join(map(str,pred_label[i]))
            f.write(peptide+'\t'+i_label+'\n')
def main(args,data):
    if args.is_train==1:
        model_train(args)
    elif args.is_train==2:
        test(args)
    else:
        model_train(args)
        test(args)

if __name__ == '__main__':
    args=parse_args()
    
    data=GetData(args.intensity_num_label)

    main(args,data)
    