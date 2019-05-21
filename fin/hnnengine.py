# This is the controller code

import importlib
import os

if __name__ == "__main__":
    mods = [m for m in os.listdir("kernels") if m.endswith(".py")]
    kers = {m[:-3]:importlib.import_module("kernels."+m[:-3]) for m in mods}
    kernels = {k:v.__dict__[k] for k,v in kers.items() if k in v.__dict__}
    for k,v in kernels.items():
        print(k,v)
        v()
