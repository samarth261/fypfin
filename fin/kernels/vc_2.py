import random
import math
from random import shuffle

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

def vc_2(G):
    vc=[]
    for i in range(0,len(G)):
        vc.append(1)
    flag=1
    vcn=[]
    while(flag>0):
        vcn=compare_nbrs(G,vc)
        if(vcn==vc):
            flag=0
        else:
            vc=vcn[:]
    return vc
