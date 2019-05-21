# This file provides a functions which accepts a list of values and returns the best result

def evaluate_partition(G,S):
    nw=0
    tw=0
    n=len(G)
    for i in range(n):
        for j in range(n):
            nw=nw+G[i][j]
            if(S[i]!=S[j]):
                tw=tw+G[i][j]
    c=tw/nw
    return tw


def choose_best(l,g, *args):
    return max([evaluate_partition(g,ii) for ii in l])

