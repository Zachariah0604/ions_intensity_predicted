import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import torch
import torch.nn as n
import torch.utils.data as td
from torch.autograd import Variable
from model import *
import argparse
from tools.get_data import *
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-train', type=int, default=0,help='1=train, 2=test, 0=train and test')
    parser.add_argument('--num-epochs', type=int, default=30,help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=64,help='the number of peptide')
    parser.add_argument('--learning-rate', type=float, default=1e-3,help='')
    parser.add_argument('--input-size', type=int, default=88,help='')
    parser.add_argument('--hidden-size', type=int, default=256,help='')
    parser.add_argument('--num-layers', type=int, default=2,help='')
    parser.add_argument('--positive_ratio', type=float, default=2.0,help='')
    parser.add_argument('--num-classes', type=int, default=4,help='')
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
    
   
    sorted_batch_peptide=[];sorted_seq_length=[]
    for j in range(_batch_number):
        _batch_peptide =batch_peptide[j]
        _seq_length =seq_length[j]
        data_sorted=[(lengths,peptides ) for lengths,peptides  in zip(_seq_length,_batch_peptide)] 
        data_sorted.sort(reverse=True) 
        sorted_length=[lengths for lengths,peptides  in data_sorted] 
        sorted_peptides=[peptides for lengths,peptides  in data_sorted]
        sorted_batch_peptide.append(sorted_peptides)
        sorted_seq_length.append(sorted_length)
    return sorted_batch_peptide,_batch_number,sorted_seq_length
   
def padding_data(data,max_ions_number):
    _ydim=data.shape[1]
    dv=max_ions_number-data.shape[0]
    data=data.tolist()
    if dv > 0:
        data.extend(np.zeros((dv,_ydim)).tolist())
        
    return data

def calc_acc(prediction_sig,max_ions_number,seq_length,y,threshold=0.5):
    prediction_numpy=prediction_sig.cpu().data.numpy()
    prediction_numpy[prediction_numpy>threshold]=1
    prediction_numpy[prediction_numpy<=threshold]=0
    mask = (np.expand_dims(np.arange(max_ions_number), axis=0) < np.expand_dims(seq_length, axis=1))
    total_labels = np.sum(seq_length)*4
    #correct_labels = np.sum((np.array(y).reshape(-1,4)[mask.reshape(1,-1).tolist()] == prediction_numpy.reshape(-1,4)[mask.reshape(1,-1).tolist()]))
    correct_labels = np.sum((np.array(y).reshape(-1,4)== prediction_numpy.reshape(-1,4))*mask.reshape(-1,1))
    accuracy = 100.0 * correct_labels / float(total_labels)
    return accuracy,prediction_numpy,mask


def train_model(args):
    #train data
    _,_,train_X,train_y,merge_train_list=data.get_discretization_data('data/data_mm/test_test.txt',2)
    print(str(len(merge_train_list[0]))+' train peptides ,DataShape:('+str(np.array(train_X).shape)+str(np.array(train_y).shape)+')')
    train_batch_peptide,train_batch_number,train_seq_length=get_batch_peptide(merge_train_list,args.batch_size)
  

    #val data
    _,_,test_X,test_y,merge_test_list=data.get_discretization_data('data/data_mm/test_test.txt',2)
    print(str(len(merge_test_list[0]))+' test peptides ,DataShape:('+str(np.array(test_X).shape)+str(np.array(test_y).shape)+')')
    test_batch_peptide,test_batch_number,test_seq_length=get_batch_peptide(merge_test_list,args.batch_size)

    print('..training model')
    model = BiLSTM(args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  
    class_weight = Variable(torch.FloatTensor([1.0, args.positive_ratio])).cuda()
    best_acc=0.0
    for epoch in range(args.num_epochs):
        train_loss=0.0;train_accuracy=0.0;val_loss=0.0;val_accuracy=0.0

        permutation_batch = np.random.permutation(len(train_batch_peptide))
        suffled_train_batch_peptide=np.array(train_batch_peptide)[permutation_batch].tolist()
        suffled_train_seq_length=np.array(train_seq_length)[permutation_batch].tolist()

        for i,peptides in enumerate(suffled_train_batch_peptide):
            
            X=[];y=[];
            max_ions_number=max(suffled_train_seq_length[i])
            for j in range(len(peptides)):
                ions=data.get_split_list(peptides[j])
                X.append(padding_data(train_X[np.array(ions)],max_ions_number))
                y.append(padding_data(train_y[np.array(ions)],max_ions_number))
         
            X=Variable(torch.FloatTensor(X).cuda())
            label=Variable(torch.FloatTensor(y).cuda())
            prediction= model(X,suffled_train_seq_length[i])
            prediction=torch.nn.functional.sigmoid(prediction)
            
            train_accuracy+= calc_acc(prediction,max_ions_number,suffled_train_seq_length[i],y)[0]
            
            weights=class_weight[label.long().view(-1,)].view(label.size())
            loss_func = nn.BCELoss(size_average=True,weight=weights)
            loss = loss_func(prediction, label)
            train_loss+=loss.data[0]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ###########val#############
        for i,peptides in enumerate(test_batch_peptide):
            ions=data.get_split_list2(peptides)
            X=[];y=[];
            max_ions_number=max(test_seq_length[i])
            for j in range(len(peptides)):
                ions=data.get_split_list(peptides[j])
                X.append(padding_data(test_X[np.array(ions)],max_ions_number))
                y.append(padding_data(test_y[np.array(ions)],max_ions_number))
          
            X=Variable(torch.FloatTensor(X).cuda())
            label=Variable(torch.FloatTensor(y).cuda())
            prediction= model(X,test_seq_length[i])
            prediction=torch.nn.functional.sigmoid(prediction)

            val_accuracy+= calc_acc(prediction,max_ions_number,test_seq_length[i],y)[0]

            weights=class_weight[label.long().view(-1,)].view(label.size())
            loss_func = nn.BCELoss(size_average=True,weight=weights)
            loss = loss_func(prediction, label)
            val_loss+=loss.data[0]
            
        ###########val#############
            #if step%100 ==0:
        print ('Epoch [%d/%d],Train Loss: %.4f Train Acc:%.3f,val Loss: %.4f val Acc:%.3f'
                    %(epoch+1, args.num_epochs,train_loss/train_batch_number,train_accuracy/train_batch_number,val_loss/test_batch_number,val_accuracy/test_batch_number,))
        #if (val_accuracy/test_batch_number)>best_acc:
        #    best_acc=val_accuracy/test_batch_number
           
        #torch.save(model,'seq2seq-WGAN/model/pytorch_model.pkl')
    print('..training complete')
    torch.save(model,'seq2seq-WGAN/model/pytorch_model.pkl')
    
    

def model_predict(args):
    print('..predicting')
    #test data
    peptide_idxs,test_peptides,test_X,test_y,merge_test_list=data.get_discretization_data('data/data_mm/test.txt',2)
    print(str(len(merge_test_list[0]))+' test peptides ,DataShape:('+str(np.array(test_X).shape)+str(np.array(test_y).shape)+')')
    
    test_batch_peptide,test_batch_number,test_seq_length=get_batch_peptide(merge_test_list,args.batch_size)
    #predict
    model=torch.load('seq2seq-WGAN/model/pytorch_model.pkl')
    class_weight = Variable(torch.FloatTensor([1.0, args.positive_ratio])).cuda() 
    test_accuracy=0.0;test_loss=0.0;test_pred_label=[];ions_index=[]
    
    for i,peptides in enumerate(test_batch_peptide):
            ions=data.get_split_list2(peptides)
            X=[];y=[];
            max_ions_number=max(test_seq_length[i])
            for j in range(len(peptides)):
                ions=data.get_split_list(peptides[j])
                ions_index.extend(ions)
                X.append(padding_data(test_X[np.array(ions)],max_ions_number))
                y.append(padding_data(test_y[np.array(ions)],max_ions_number))
          
            X=Variable(torch.FloatTensor(X).cuda())
            label=Variable(torch.FloatTensor(y).cuda())
        
            prediction= model(X,test_seq_length[i])
            prediction_sig=torch.nn.functional.sigmoid(prediction)

            accuracy,prediction_numpy,mask=calc_acc(prediction_sig,max_ions_number,test_seq_length[i],y)
            test_accuracy+= accuracy

            prediction_numpy=prediction_numpy.reshape(-1,4)[mask.reshape(1,-1).tolist()]
            test_pred_label.extend(prediction_numpy.tolist())
            #real.extend(np.array(y).reshape(-1,4)[mask.reshape(1,-1).tolist()].tolist())
           
            weights=class_weight[label.long().view(-1,)].view(label.size())
            loss_func = nn.BCELoss(size_average=True,weight=weights)
            loss = loss_func(prediction_sig, label)
            test_loss+=loss.data[0]
    
   
   
    print ('test Loss: %.4f test Acc:%.3f' %(test_loss/test_batch_number,test_accuracy/test_batch_number,)) 
    return test_pred_label,ions_index


def test(args):
    test_pred_labels,ions_index=model_predict(args)
    data_sorted=[(ion_index,test_pred_label ) for ion_index,test_pred_label  in zip(ions_index,test_pred_labels)] 
    data_sorted.sort() 
    sorted_ions_index=[ion_index for ion_index,test_pred_label  in data_sorted] 
    sorted_test_pred_labels=[test_pred_label for ion_index,test_pred_label  in data_sorted]


    with open('data/data_mm/torch_classfication_test.txt','w') as f:
        for i in range(len(sorted_test_pred_labels)):
            
            i_label=','.join(map(str,sorted_test_pred_labels[i]))
            f.write(str(sorted_ions_index[i])+'\t'+i_label+'\n')
    
    #with open('data/data_mm/real.txt','w') as rf:
    #    for k in range(len(sorted_reals)):
    #        
    #        k_label=','.join(map(str,sorted_reals[k]))
    #        rf.write(str(sorted_ions_index[k])+'\t'+k_label+'\n')
def main(args):
    if(args.is_train==1):
        train_model(args)
    elif(args.is_train==2):
        test(args)
    else:
        train_model(args)
        test(args)

if __name__ == '__main__':
    data=GetData(4)
    main(parse_args())

   







