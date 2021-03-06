import random
import math
from random import shuffle
import time
def generate_graph(n,m):
    G=[]
    #initialize the adjacency matrix to null state
    for x in range(n):
        temp=[]
        for y in range(n):
            temp.append(0)
        G.append(temp)
    #assigning weights to randomly created edges
    for a in range(m):
        i=random.randint(0,n-1)
        j=random.randint(0,n-1)
        while(G[i][j]>0 or i==j):
            i=random.randint(0,n-1)
            j=random.randint(0,n-1)
        G[i][j]=1
        G[j][i]=1
    return G
class params:
    A=2.5
    B=0.5
    beta=0.05
def sigmoid(x,theta=5):
    y=1/(1+(math.pow(math.e,(-x/theta))))
    return y
def temperature(t,n):
    t0=0.15*n
    t=math.pow((1-params.beta),t)*t0
    return t
def vc_1(G):
    m=1
    for i in G:
        for j in i:
            if (j==1):
                m=m+1
    m=m/2
    d=G
    u=[]
    v=[]
    vd=[]
    for i in range(0,len(d)):
        random.seed(int(time.time()))
        r=random.randint(0,2147483647)
        h=0.5+((r%11)/100)-0.05
        u.append(h)
    for i in range(0,len(u)):
        v.append(sigmoid(u[i]))
       
    energies=[]
    for ite in range(0,1000):
        temp=temperature(ite,len(G))
        for i in range(0,len(G)):
            t1=-params.A
            s1=0
            for j in range(0,len(G)):
                s1=s1+(d[i][j]*v[j])
            t2=(-2)*params.B*s1
            s2=0
            for j in range(0,len(G)):
                s2=s2+d[i][j]
            t3=(2)*params.B*s2
            t4=(-1)*temp*(1-(s2/m))
            u[i]=t1+t2+t3+t4
            v[i]=sigmoid(u[i])
    for i in range(0,len(v)):
        if(v[i]>=0.5):
            vd.append(1)
        else:
            vd.append(0)
    return vd
