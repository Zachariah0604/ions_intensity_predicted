import os
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
if __name__ =='__main__':
    files,names=get_files('D:\\data\\mgf')
    cunt=0
  
    charge_1_cunt=0 
    charge_2_cunt=0 
    charge_3_cunt=0 
    charge_4_cunt=0 
    charge_5_cunt=0 
    charge_6_cunt=0 
    charge_7_cunt=0 
    charge_8_cunt=0 
    for file in files:
        with open(file,'r') as f:
            while True:
                line=f.readline()
                if not line:
                    break
                if 'BEGIN' in line:
                    cunt+=1
                if 'CHARGE=' in line:
                    _charge=int(line.split('=')[1].replace('+','').replace('\n',''))
                    if _charge==1:
                        charge_1_cunt+=1
                    if _charge==2:
                        charge_2_cunt+=1
                    if _charge==3:
                        charge_3_cunt+=1
                    if _charge==4:
                        charge_4_cunt+=1
                    if _charge==5:
                        charge_5_cunt+=1
                    if _charge==6:
                        charge_6_cunt+=1
                    if _charge==7:
                        charge_7_cunt+=1
                    if _charge==8:
                        charge_8_cunt+=1
                #if 'Acety' in line:
                #    a_cunt+=1
    print(cunt)
    print(charge_1_cunt)
    print(charge_2_cunt)
    print(charge_3_cunt)
    print(charge_4_cunt)
    print(charge_5_cunt)
    print(charge_6_cunt)
    print(charge_7_cunt)
    print(charge_8_cunt)