#!/usr/bin/python3.5

# This is the controller code

import importlib
import os
import pickle


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

def load_all_forward_reduction_functions():
    mods = [m for m in os.listdir("reductions") if m.endswith("forward.py")]
    reds = {}
    for m in mods:
        # basically ignoring the " forward.py" part
        jj = (m[:-len(" forward.py")]).split(" ")
        reds[tuple(jj)] = importlib.import_module("reductions." + m[:-3]).convert
    
    return reds

def load_all_reverse_reduction_functions():
    mods = [m for m in os.listdir("reductions") if m.endswith("reverse.py")]
    reds = {}
    for m in mods:
        # basically ignoring the " forward.py" part
        jj = (m[:-len(" reverse.py")]).split(" ")
        reds[tuple(jj)] = importlib.import_module("reductions." + m[:-3]).convert
    
    return reds



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
    reductions = load_all_reductions() # This only has the mappings from what to what
    forward_reduction_functions = load_all_forward_reduction_functions()
    reverse_reduction_functions = load_all_reverse_reduction_functions()

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
    for kernerl_name in kernels.keys():
        if kernerl_name.startswith(cmdlineargs.problem + "_"):
            direct_solving_kernels.append((kernerl_name, kernels[kernerl_name]))
    
    print("direct kerels", direct_solving_kernels)

    # list of tuples(str,str)
    applicable_reductions = [(a,b) for a,b in reductions if a == cmdlineargs.problem]
    kernels_after_reduction = []

    if not cmdlineargs.nored:
        for f,t in applicable_reductions:
            for kern_name, fxobject in kernels.items():
                if kern_name.startswith(t + "_"):
                    kernels_after_reduction.append((kern_name, fxobject))
    print(reductions)
    print(applicable_reductions)
    print("kernel after reduction", kernels_after_reduction)

    # This list will have all the solutions from the different reductions
    solutions = []
    # First we get solutions by using direct kernels
    for kern_name, fxobject in direct_solving_kernels:
        print("ker name", kern_name)
        solutions.append((kern_name, fxobject(pickle.load(open(cmdlineargs.file, "rb")))))
    
    for kern_name, fxobject in kernels_after_reduction:
        result = fxobject(
            forward_reduction_functions[(cmdlineargs.problem, kern_name.split("_")[0])](
                pickle.load(open(cmdlineargs.file, "rb"))
            )
        )
        solutions.append(
            (
            kern_name, 
            reverse_reduction_functions[(cmdlineargs.problem, kern_name.split("_")[0])](result)
            )
        )

    print(solutions)
    # Finally we choose the best result from amongst many
    resolver = importlib.import_module("selector." + cmdlineargs.problem).choose_best
    final_answer = resolver([ii for jj,ii in solutions])
    print(final_answer)


