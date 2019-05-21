# This file provides a functions which accepts a list of values and returns the best result

def choose_best(l):
    return max([sum(ii) for ii in l])

