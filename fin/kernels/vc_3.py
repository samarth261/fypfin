import random
from random import shuffle
import time
import math
def produce_neighbours(v):
    nbrs=[]
    for i in range(0,len(v)):
        u=v[:]
        u[i]=1-v[i]
        nbrs.append(u)
    return nbrs
def vc_check(G,vc):
    for i in range(0,len(G)):
        for j in range(0,len(G)):
            if((G[i][j]==1) and ((vc[i]==0) and (vc[j]==0))):
                return 0
    return 1
                    
            
    
            
def compare_nbrs(G,vc):
    vcn=vc[:]
    nbrs=produce_neighbours(vc)
    random.seed(time.time())
    shuffle(nbrs)
    for v in nbrs:
        if((sum(v)<sum(vcn)) and (vc_check(G,v)==1)):
            vcn=v[:]
    return vcn

def compare_nbrs_sa(G,vc,t):
    vcn=vc[:]
    nbrs=produce_neighbours(vc)
    random.seed(time.time())
    shuffle(nbrs)
    for v in nbrs:
        if((sum(v)<sum(vcn)) and (vc_check(G,v)==1)):
            vcn=v[:]
        elif((vc_check(G,v)==1) and t>=1):
            random.seed(time.time())
            if(random.random()<math.pow(math.e,((sum(vcn)-sum(v))/t))):
               vcn=v[:]
    return vc
def vc_3(G):
    T=50
    vc=[]
    for i in range(0,len(G)):
        vc.append(1)
    vcn=[]
    flag=1
    while(flag>0):
        vcn=compare_nbrs_sa(G,vc,T)
        if(vcn==vc):
            flag=0
        else:
            vc=vcn[:]
        T=T-1
    return vc
               
            

