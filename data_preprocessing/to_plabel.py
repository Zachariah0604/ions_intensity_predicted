import os
import pandas as pd
import numpy as np

def get_merge_list(data):
    #Number,Intensity
    temp_peptide = ''
    temp_list = []
    peptide_list = []
    temp_intensity=[]
    temp_ion_type=[]
    temp_r_intensity=[]
    #print(data)
    for row in data.itertuples():
        _peptide = row.peptide
        _intensity=row.intensity
        _charge=row.charge
        _modi=row.modi
        if _peptide != temp_peptide:
            if temp_peptide == '':
                temp_r_intensity.append(str(row.r_intensity))
                temp_intensity.append(str(_intensity))
                aaaa=row.ion_type
                temp_ion_type.append(row.ion_type)
                temp_list.append([str(_charge),_peptide.upper(),_modi,temp_ion_type,temp_intensity,temp_r_intensity])
            else:
                peptide_list.append(temp_list[0])
                temp_list = []
                temp_intensity=[]
                temp_ion_type=[]
                temp_r_intensity=[]
                temp_r_intensity.append(str(row.r_intensity))
                temp_intensity.append(str(_intensity))
                temp_ion_type.append(row.ion_type)
                temp_list.append([str(_charge),_peptide.upper(),_modi,temp_ion_type,temp_intensity,temp_r_intensity])
            temp_peptide = _peptide
        else:
            temp_r_intensity.append(str(row.r_intensity))
            temp_intensity.append(str(_intensity))
            temp_ion_type.append(row.ion_type)
            
    peptide_list.append(temp_list[0])
    
    return peptide_list

def main():
    file='data/data_proteometools/NCE35_b_y_train.txt'
   
    data = pd.read_table(file,header=None,names=['peptide','charge','ion','modi','real_mass','ion_type','r_intensity','intensity'],index_col=False,na_values=['asdasdefesefd'],keep_default_na=False)
    print('DataShape: ' + str(data.shape))
    merge_list=get_merge_list(data) 
    with open('data/data_for_pdeep_test/proteometools_NCE35_b_y_train.plabel','w') as wf:
        pass
    for _list in merge_list:
        with open('data/data_for_pdeep_test/proteometools_NCE35_b_y_train.plabel','a') as wf:
            _modi=_list[2]
            if _modi=='NULL':
                _modi=''
            else:
                _modi=_list[2].replace('],','];')
            ion_types=','.join(_list[3]).replace('++','+2').replace('+,','+1,')
            
            #if max(list(map(float,_list[4])))>0:
            wf.write(str(_list[0])+'\t'+_list[1]+'\t'+_modi+'\t'+ion_types+','+'\t'+','.join(_list[4])+','+'\t'+','.join(_list[5])+',' +'\n')
           
if __name__=='__main__':
    main()