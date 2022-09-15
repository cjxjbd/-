import os
import os.path as osp
import re
import random
import shutil
from tqdm import tqdm

root_path='/home/zyp/pythonProject/L2CS-Net-main/datasets/MPIIFaceGaze'
ori='Label'
train='train'
val='val'
test='test'
ratio=(6,2,2) #zhonghe =10
is_shuffle=True

datasets=(train,val,test)
def unit(root_path,ori,ratio,datasets,is_shuffle):

    ori=osp.join(root_path,ori)
    for d_type in datasets:
        outpath=osp.join(root_path,d_type)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    tr = (0,ratio[0]/10)
    va = (ratio[0]/10,(ratio[0]+ratio[1])/10)
    te = ((ratio[0]+ratio[1])/10,1)

    filelist=[f for f in os.listdir(ori) if re.search('.*[label]$',f)]
    filelist=sorted(filelist,key=lambda f:int(f[1:3]))
    #print(filelist)
    for f in filelist:
        path=osp.join(ori,f)
        colletlist=[]
        with open(path,'r',encoding='utf-8') as lab:
            colletlist,fir=readlab(lab,colletlist,is_shuffle)

        outpath = osp.join(root_path, datasets[0])
        lent=len(colletlist)
        with open(osp.join(outpath,f),'w',encoding='utf-8') as lab:
            writelab(lab, colletlist[int(lent*tr[0]):int(lent*tr[1])],fir)
        outpath = osp.join(root_path, datasets[1])
        with open(osp.join(outpath,f),'w',encoding='utf-8') as lab:
            writelab(lab, colletlist[int(lent*va[0]):int(lent*va[1])],fir)
        outpath = osp.join(root_path, datasets[2])
        with open(osp.join(outpath,f),'w',encoding='utf-8') as lab:
            writelab(lab, colletlist[int(lent*te[0]):int(lent*te[1])],fir)

def writelab(lab, c_list,fir):
    lab.writelines(fir)
    for line in c_list:
        lab.writelines(line)

def readlab(lab,colletlist,is_shuffle):
    line=lab.readline()
    while line:
        if line[0]=='F':
            fir=line
        elif line[0]=='p':
            colletlist.append(line)
        else:
            break
        line = lab.readline()
    #print(colletlist)
    if is_shuffle:
        random.shuffle(colletlist)

    return colletlist,fir

unit(root_path,ori,ratio,datasets,is_shuffle)

