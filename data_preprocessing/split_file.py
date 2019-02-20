import random
import pandas as pd
f_train=open('data/data_mm/train.txt','w')
f_test=open('data/data_mm/test.txt','w')
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
                temp_list.append([_peptide,str(_charge),row.ion,_modi,temp_ion_type,temp_intensity,temp_r_intensity])
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
if __name__=='__main__':
    file='data/data_mm/test_test.txt'
    data = pd.read_table(file,header=None,names=['peptide','charge','ion','modi','real_mass','ion_type','r_intensity','intensity'],index_col=False,na_values=['asdasdefesefd'],keep_default_na=False)
    print('DataShape: ' + str(data.shape))
    merge_list=get_merge_list(data) 
    for _list in merge_list:
        a=random.random()
        if a< 0.8:
            for i in range(len(_list[4])):
                f_train.write(_list[i])
        else:
            for i in range(len(_list[4])):
                f_test.write(_list[i])

f_train.close()
f_test.close()
