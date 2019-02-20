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
    ions=[]
    ion_mass=[]
    #print(data)
    for row in data.itertuples():
        _peptide = row.peptide
        
        _intensity=row.intensity
        if _intensity==-1.0:
            _intensity=0.0
        
        _charge=row.charge
        _modi=row.modi
        if _peptide != temp_peptide:
            if temp_peptide == '':
                temp_r_intensity.append(str(row.r_intensity))
                temp_intensity.append(str(_intensity))
                aaaa=row.ion_type
                temp_ion_type.append(row.ion_type)
                ions.append(str(row.ion))
                ion_mass.append(str(row.real_mass))
                temp_list.append([str(_charge),_peptide.upper(),ions,_modi,ion_mass,temp_ion_type,temp_intensity,temp_r_intensity])
            else:
                peptide_list.append(temp_list[0])
                temp_list = []
                temp_intensity=[]
                temp_ion_type=[]
                temp_r_intensity=[]
                ions=[]
                ion_mass=[]

                temp_r_intensity.append(str(row.r_intensity))
                temp_intensity.append(str(_intensity))
                temp_ion_type.append(row.ion_type)
                ions.append(str(row.ion))
                ion_mass.append(str(row.real_mass))
                temp_list.append([str(_charge),_peptide.upper(),ions,_modi,ion_mass,temp_ion_type,temp_intensity,temp_r_intensity])
            temp_peptide = _peptide
        else:
            temp_r_intensity.append(str(row.r_intensity))
            temp_intensity.append(str(_intensity))
            temp_ion_type.append(row.ion_type)
            ions.append(str(row.ion))
            ion_mass.append(str(row.real_mass))
            
    peptide_list.append(temp_list[0])
    
    return peptide_list
def get_files(rootdir):
        files=[]
        names=[]
        _list=os.listdir(rootdir)
        for i in range(0,len(_list)):
            path=os.path.join(rootdir,_list[i])
            if os.path.isfile(path):
                files.append(path)
                names.append(_list[i])
        return files,names
def main():
    files,names=get_files('data/data_swedcad_mm/am')
    for file_index,file in enumerate(files):
        if 'train_.txt' in file:
            data = pd.read_table(file,header=None,names=['peptide','charge','ion','modi','real_mass','ion_type','r_intensity','intensity','max_intensity'],index_col=False,na_values=['asdasdefesefd'],keep_default_na=False)
            print(data.loc[1946,:])
            print('DataShape: ' + str(data.shape))
            merge_list=get_merge_list(data) 
            with open('data/data_swedcad_mm/'+names[file_index].split('.')[0]+'1label.txt','w') as wf:
                pass
            for _list in merge_list:
                with open('data/data_swedcad_mm/am/'+names[file_index].split('.')[0]+'1label.txt','a') as wf:
                    _charge=_list[0]
                    _peptide=_list[1]
                    _ions=_list[2]

                    _modi=_list[3]
                    if pd.isnull(_modi):
                        _modi=''
                    _ion_mass=_list[4]
                    _ion_type=_list[5]
                    _intensity=_list[6]
                    _r_intensity=_list[7]

                    for _i in range(1,len(_peptide)):
                        b_ion_type='b'+str(_i)
                        y_ion_type='y'+str(len(_peptide)-_i)

                        b_pos=-1;y_pos=-1;
                        try:
                            b_pos=_ion_type.index(b_ion_type)
                        except:
                            pass
                        try:
                            y_pos=_ion_type.index(y_ion_type)
                        except:
                            pass
                        b_ion='';b_mass='0.0';b_inten='0.0';b_r_inten='0.0'
                        y_ion='';y_mass='0.0';y_inten='0.0';y_r_inten='0.0'
                        if b_pos!=-1:
                            b_ion=_ions[b_pos]
                            b_mass=_ion_mass[b_pos]
                            b_inten=_intensity[b_pos]
                            b_r_inten=_r_intensity[b_pos]
                        else:
                            b_ion=_peptide[:_i]
                        if y_pos!=-1:
                            y_ion=_ions[y_pos]
                            y_mass=_ion_mass[y_pos]
                            y_inten=_intensity[y_pos]
                            y_r_inten=_r_intensity[y_pos]
                        else:
                            y_ion=_peptide[_i:]

                        #wf.write(str(_peptide)+'\t'+str(_charge)+'\t'+str(b_ion+','+y_ion)+'\t'+_modi+'\t'+str(b_mass+','+y_mass)+'\t'+str(b_ion_type+','+y_ion_type)+'\t'+str(b_r_inten+','+y_r_inten)+'\t'+str(b_inten+','+y_inten)+'\n')
                        wf.write(str(_peptide)+'\t'+str(_charge)+'\t'+b_ion+'\t'+_modi+'\t'+str(b_mass)+'\t'+str(b_ion_type)+'\t'+str(b_r_inten)+'\t'+str(b_inten)+'\n')
                        wf.write(str(_peptide)+'\t'+str(_charge)+'\t'+y_ion+'\t'+_modi+'\t'+str(y_mass)+'\t'+str(y_ion_type)+'\t'+str(y_r_inten)+'\t'+str(y_inten)+'\n')
           
if __name__=='__main__':
    main()