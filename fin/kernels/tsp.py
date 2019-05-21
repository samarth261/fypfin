# This file defines the tsp function

def fx1():
    print("supporting tsp")
    return "tsp"

def tsp(*args, **kwargs):
    print("This is the tsp kernel")
    print("using fx1")
    s = fx1()
    print("returned", s)

if __name__=="__main__":
    print("running independently")
    tsp()


