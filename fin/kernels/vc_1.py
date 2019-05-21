import random
import math
from random import shuffle
class params:
    A=1
    B=0.8
    beta=0.05
def sigmoid(x,theta=5):
    y=1/(1+(math.pow(math.e,(-x/theta))))
    return y
def temperature(t,n):
    t0=0.15*n
    t=math.pow((1-params.beta),t)*t0
    return t
def vc_1(G,m):
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
    for ite in range(0,5000):
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
        print(v)
    for i in range(0,len(v)):
        if(v[i]>=0.5):
            vd.append(1)
        else:
            vd.append(0)
