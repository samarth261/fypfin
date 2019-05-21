### Setting Up

Run the install.sh   `bash install.sh`

#### Prerequisites:

python3.5 should be installed

### Running the software

`python3 hnnengine.py --file <input file name> --problem <vc|mc|tsp> [--nored]`

---

#### Output of `pytohn hnnengine.py -h`

usage: hnnengine.py [-h] [--file FILE] [--problem PROBLEM] [--nored]

optional arguments:
  -h, --help         show this help message and exit
  --file FILE        The input problem file
  --problem PROBLEM  Specifies the type of the problem whether tsp or vc or so
                     on..
  --nored            Setting this flag will force the engine to not use any
                     sort of reduction for solving the problem.

The `<input file>` should be a serialized format of the input graph.

----

Also the kernels can be run individually by accessing them from the `kernels` folder.

---

- New reductions can be added by making an entry in the `reductions/all_reductions` , at the same time two files `from to forward.py` and `from to reverse.py` should be added, each defining `convert` function.

- New kernel can be added by creating a new file `new_kernel_<n+1>.py` that defines a function `new_kernel`; where n is the previous number of kernels for that problem. The newly defined function should take exactly one argument that is the graph instance itself.
- For every new kernel add a new file `<new_kernel>.py` needs to be added into the path `./selector`. it should define a function called `choose_best`. The parameters of this are open ended.