def flip_or_not(G,v,S,e):
    wo=0
    wi=0
    for i in range(len(G)):
        if(G[v][i]>0):
            if(S[v]==S[i]):
                wi=wi+G[v][i]
            if(S[v]!=S[i]):
                wo=wo+G[v][i]
    n=len(G)
    if(wi>wo+((2*e*wo)/n)):
        return 1
    else:
        return -1
#algorithm for local search
def mc(G,e=0.1):
    n=len(G)
    S=[]
    for i in range(n):
        S.append(random.randint(0,1))
    proceed=1
    #print(S)
    while(proceed>0):
        proceed=0
        for i in range(n):
            bol=flip_or_not(G,i,S,e)
            if(bol==1):
                temp=S[i]
                S[i]=1-temp
                proceed=1
                #print(bol)
                #print(S)
                break
    return S
