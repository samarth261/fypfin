#!/usr/bin/python3.5

# This is the controller code

import importlib
import os


# Loading all the modules inside the kernels folder/package
def load_all_kernels():
    # returns a dictionary of the modules
    mods = [m for m in os.listdir("kernels") if m.endswith(".py")]
    kers = {m[:-3]:importlib.import_module("kernels."+m[:-3]) for m in mods}
    kernels = {k:v.__dict__[k] for k,v in kers.items() if k in v.__dict__}
    # for k,v in kernels.items():
    #     print(k,v)
    #     v()
    return kernels

def load_all_reductions():
    reds_file = open("reductions/all_reductions","r")
    reductions = []
    for line in reds_file.read().split():
        # f is for from and t is for to
        f,t = line.split("->")
        reductions.append((f,t))
    return reductions

def get_cmd_line_parser():
    # returns the command line argument
    import argparse
    parser = argparse.ArgumentParser()

    # Add all the cmd line arguments accepted by the hnnengine
    parser.add_argument("--file", type=str, help="The input problem file")
    parser.add_argument("--problem", type=str, help="Specifies the type of the problem whether tsp or vc or so on..")
    parser.add_argument("--nored", action='store_true', help="Setting this flag will force the engine to not use any sort of reduction for solving the problem.")

    return parser.parse_args()


if __name__ == "__main__":
    # Loading all the kernels the cost functions etc
    kernels = load_all_kernels()
    reductions = load_all_reductions()

    # Parsing the commandlien arguments
    cmdlineargs = get_cmd_line_parser()

    #---------------------------------------------
    # Check that that filename and the problem aren't None
    assert cmdlineargs.file is not None
    assert cmdlineargs.problem is not None

    # Check that the problem is in the kernels
    # assert cmdlineargs.problem in kernels.keys()

    # This list should have all the functions directly solving this
    # The elements of this are tuples with 2 elements
    # - the kernel name
    # - the function object
    direct_solving_kernels = []
    for kernerl_name, fxobject in kernels.items():
        if kernerl_name.startswith(cmdlineargs.problem + "_"):
            direct_solving_kernels.append((kernerl_name, fxobject))
    
    print("direct kerels", direct_solving_kernels)

    applicable_reductions = [(a,b) for a,b in reductions if a.startswith(cmdlineargs.problem + "_")]
    kernels_after_reduction = []
    for f,t in applicable_reductions:
        for kern_name, fxobject in kernels.items():
            if kern_name.startswith(t.split("_")[0] + "_"):
                kernels_after_reduction.append((kern_name, fxobject))
    print(reductions)
    print(applicable_reductions)
    print("kernel after reduction", kernels_after_reduction)

    # This list will have all the solutions from the different reductions
    solutions = []



