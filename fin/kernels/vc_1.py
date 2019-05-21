# This defines the vertec cover kernel

def fx1():
    print("supporting vc")
    return "vc"

def vc_1(*args, **kwargs):
    print("The vertex cover kernel")
    s = fx1()
    print("ret", s)


if __name__=="__main__":
    vc_1()

