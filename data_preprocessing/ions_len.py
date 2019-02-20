
import os
from collections import defaultdict

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
files,names=get_files('data/data_mm/')
dic = defaultdict(list)
dic_inten=defaultdict(list)
sum_of_intensity=0.0
for file_index,file in enumerate(files):
    
    if 'sum_of_intensity' in names[file_index]:
        with open(file,'r') as rf:
            while True:
                line=rf.readline()
                if not line:
                    break
                sum_of_intensity+=float(line)
    if 'ay_by_' in names[file_index]:
        for i in [1,2,3,4,5,6,7,8]:
            dic[names[file_index]+'_len'+str(i)]=0
            dic_inten[names[file_index]+'_len'+str(i)]=0.0
            dic[names[file_index]+'_len'+str(i)+'_all']=0 
        
        with open(file,'r') as f:
            while True:
                line=f.readline()
                if not line:
                    break
                _list=line.split('\t')
                _ions_len=len(_list[2])
                is_succ_list=_list[6].split(',')
                for j in range(2):
                    dic[names[file_index]+'_len'+str(_ions_len)+'_all']+=1
                    if float(is_succ_list[j]) >0.0:
                        dic_inten[names[file_index]+'_len'+str(_ions_len)]+=float(_list[7].split(',')[j])
                        dic[names[file_index]+'_len'+str(_ions_len)]+=1
                    

_dic_list=sorted(dic.items(), key=lambda d: d[0],reverse=False)
print(_dic_list)
print(dic_inten)
with open('data/data_mm/result.txt','w') as f:
    k=1
    _val=0
    for value in _dic_list:
        
        if k%2== 0:
            f.write(value[0]+':'+str(value[1])+','+str(_val/value[1])+'\n')
        else:
           f.write(value[0]+':'+str(value[1])+'\n')
        k+=1
        _val=value[1]
    for key,valu in dic_inten.items():
       
       
        f.write(key+':'+str(valu)+'/'+str(sum_of_intensity)+'='+str(valu/sum_of_intensity)+'\n')
     

