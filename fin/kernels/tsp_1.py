#!/usr/bin/python3.5
# This u(x,i) are what are in the network.
# The v(x,i) is what we get after applying gou on the u(x,i)
# v(x,i) are what are used in the update rule.

import numpy as np
import pickle
import time
from math import tanh
import copy
import tracer # imports ./tracer.py

u0 = 0.01
delta_t = 0.0001
termination_threshold = 0.1
non_interactive = False
no_plot = False

stop_at_iteration_count = False

trc = None
    
class params:
    A = 500
    B = 500
    C = 200
    D = 250


def gou(u):
    global u0
    V = 0.5*(1 + tanh(u/u0))
    return V

# we are going to generate the co-ordinates of the given number of cities
# and then return a distance matrix for the same
def rand_city(count, x_max, y_max):
    # count is the number of cities
    # max_x and max_y are the maximum allowed ranges for the cities
    
    # this will store the list of cities
    cities = [None,None]
    dist = []
    # cities[0] = np.random.random(x_max,size = count)
    # cities[1] = np.random.random(y_max,size = count)

    cities[0] = np.random.random_sample(size = count)
    cities[1] = np.random.random_sample(size = count)
    cities = np.transpose(np.array(cities))
    # now the cities.shape is (count, 2)

    # now we calculate the distance matrix
    from math import sqrt as sqrt
    def dist_fx(a,b):
        return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    for i in range(len(cities)):
        dist.append([])
        for j in range(len(cities)):
            dist[i].append([])
            dist[i][j] = dist_fx(cities[i],cities[j])

    return cities, np.array(dist)


def energy_of(network, distances, cities):
    # This function will return the Energy corresponding to the current
    # cofiguration of the network using the distances betwwen the cities
    # from the distances parameter
    E = 0 # this is energy we return

    A = params.A # for the rows
    B = params.B # for the colums
    C = params.C # for the sum of N 1s
    D = params.D # for the actual route

    num_cities = len(cities)

    vectorised_gou = np.vectorize(gou)
    
    network = vectorised_gou(network)
    # now we go on adding terms to it

    # row constraint
    row_term = 0
    for row in network:
        row_on_row = np.matrix(row.reshape(-1,1)) * np.matrix(row.reshape(1,-1))

        # swap_0_and_1 is np.array and so is mask
        swap_0_and_1 = np.vectorize(lambda x : 0 if x == 1 else 1)
        mask = swap_0_and_1(np.identity(num_cities))

        # now we remove the diagonal of the matrix
        row_on_row = row_on_row * mask

        row_term += np.sum(row_on_row)
    
    E += (A / 2.0) * (row_term)
    # print("->", E)


    # column constraint
    column_term = 0
    for col in network.transpose():
        col_on_col = np.matrix(col.reshape(-1,1)) * np.matrix(col.reshape(1,-1))

        # swap_0_and_1 is np.array and so is mask
        swap_0_and_1 = np.vectorize(lambda x : 0 if x == 1 else 1)
        mask = swap_0_and_1(np.identity(num_cities))

        # now we remove the diagonal of the matrix
        col_on_col = col_on_col * mask

        column_term += np.sum(col_on_col)

    E += (B / 2.0) * (column_term)
    # print("-->", E)

    # the n_cities term
    n_cities_term = network.sum() - num_cities
    n_cities_term **= 2
    E += (C / 2.0 * n_cities_term)
    # print("--->", E)

    # the distance constraint
    total_distance = 0
    for x in range(num_cities):
        for y in range(num_cities):
            if y == x:
                continue
            for i in range(num_cities):
                total_distance += ( distances[x][y] * network[x][i] * 
                                (network[y][(i+1)%num_cities] + network[y][(i-1)%num_cities]) )

    E += (D / 2.0) * total_distance
    # print("---->", E)
    return E


def delta_uxi(network, distances, cities):
    # returns a matrix of the same shape as the netwotk and it has the deltas
    # to be add to the neurons.
    
    tou = 1
    num_cities = len(cities)

    A = params.A
    B = params.B
    C = params.C
    D = params.D

    vectorised_gou = np.vectorize(gou)
    V = vectorised_gou(network)

    delta_network = np.zeros((num_cities, num_cities))

    delta_network = np.array(network)
    delta_network = -delta_network / tou
    for x in range(num_cities):
        for i in range(num_cities):
            # along the row
            row_term = 0
            for j in range(num_cities):
                if i == j:
                    continue
                row_term += V[x][j]
            delta_network[x][i] -= A * row_term

            # along column
            column_term = 0
            for y in range(num_cities):
                if y == x:
                    continue
                column_term += V[y][i]
            delta_network[x][i] -= B * column_term

            # total count term
            # count_term = 0
            # for X in range(num_cities):
            #     for j in range(num_cities):
            #         count_term += V[X][j]
            count_term = np.sum(V)
            
            delta_network[x][i] -= C * (count_term - num_cities)

            # distance term
            distance_term = 0
            for y in range(num_cities):
                distance_term += distances[x][y] * (
                    V[y][(i+1)%num_cities]+
                    V[y][(i-1)%num_cities])
            delta_network[x][i] -= D * distance_term
    
    return delta_network

def hopfield_tsp(cities, distances, target_axes, iteration_count):
    success_flag = False
    num_cities = len(cities)
    network = (np.random.rand(num_cities, num_cities)-0.5)*2
    network = pickle.load(open('tracers/f31bbcae-8b93-41c2-8bca-2e5943d263d0','rb')).init_net

    cached_initial_network = copy.copy(network)
    # energy_line = target_axes.plot([0],[energy_of(network, distances, cities)])
    if not no_plot:
        energy_line = target_axes.plot([],[])[0]
    xdata = []
    ydata = []

    threshold_function = lambda x : 1 if x > termination_threshold else 0
    threshold_function = np.vectorize(threshold_function)

    vec_gou = np.vectorize(gou)

    def all_zero(a):
        # Takes in an numpy.ndarray and returns true if all the elements are
        for ii in a:
            if 1 != ii:
                return False
        return True
    
    old_energy = energy_of(network, distances, cities)
    true_iter = 0
    while True:
        for iteration in range(iteration_count): #*num_cities):
            row_flag = False
            column_flag = False
            # print(network)
            delta = delta_uxi(network, distances, cities)
            network += (delta * delta_t)
            energy = energy_of(network, distances, cities)
            print ("Iteration", iteration, "np.sum:", np.sum(threshold_function(vec_gou(network))), "Energy:", np.round(energy,5), end="\r")

            # Now we look for the termination condition
            # The one suggest in the paper is to use the threshold value of 0.1
            after_threshold = threshold_function(vec_gou(network))
            if all_zero(np.sum(after_threshold, 0)):
                # Along column all zero
                # print("Along column all zero but for 1")
                column_flag = True
            
            if all_zero(np.sum(after_threshold, 1)):
                # Along row all zero
                # print("Along row all zero but for 1")
                row_flag = True
            
            if column_flag and row_flag:
                # This is the success condition
                total_distance = 0
                for x in range(num_cities):
                    for y in range(num_cities):
                        if y == x:
                            continue
                        for i in range(num_cities):
                            total_distance += ( distances[x][y] * after_threshold[x][i] * 
                                            (after_threshold[y][(i+1)%num_cities] + after_threshold[y][(i-1)%num_cities]) )
                
                print("success total distance =", total_distance)
                success_flag = True
                trc.add_solution(after_threshold)
                break
                
            if abs(old_energy - energy) < 10**-35:
                # Abrupt end
                print("\ntoo less")
                # break
                # print("\n", np.around(vec_gou(network), decimals=2))
                return (False,None)

            old_energy = energy


            # This is for the graph being generated
            xdata.append(true_iter+iteration)
            ydata.append(energy)
            # energy_line.set_xdata(list(energy_line.get_xdata)+[iteration])
            # energy_line.set_ydata(list(energy_line.get_ydata)+[energy])
            # we get the energy then the deltas to be added
            # we make the changes and then iterate.
            # the network has the u(x,i)
        true_iter = len(ydata)

        # Now we set the x and y lim of the plot
        if not no_plot:
            energy_line.set_xdata(xdata)
            energy_line.set_ydata(ydata)
            target_axes.set_xlim(left = 0, right = true_iter)
            target_axes.set_ylim(bottom=min(energy_line.get_ydata()) , top=max(energy_line.get_ydata()))
            print("\n", np.around(vec_gou(network), decimals=2))

        if success_flag:
            trc.set_true_itercnt_and_init_network(true_iter, cached_initial_network)
            return (True,total_distance,after_threshold)

        
        if not non_interactive:
            ch = input()
            if ch != "y":
                break
        else:
            time.sleep(1)
        
        if stop_at_iteration_count:
            print("Not retrying")
            return (False, None)
    
def get_cmdline_args():
    # returns a parser required to parse the command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str)
    parser.add_argument("-n", "--ncity", type=int, default=10)
    parser.add_argument("--delta_t", type=float, default=0.00001, help="This is the analogue of the step size/learning rate")
    parser.add_argument("--u0", type=float, default=0.01, help="This is the gain value. It decides how drastic the activation function is. A lower value implies a more drastic curve.")
    parser.add_argument("--itercnt", type=int, default=100, help="This is the number of iterations that is run before we ask if we wish to continue or not.")
    parser.add_argument("--termthresh", type=float, default=0.1, help="This is the threshold value used for seeing if that city has to be used for that location in the tour or not")
    parser.add_argument("--nonint", action='store_true', help="Set this flag so that continuously keep runnning till the termination is met")
    parser.add_argument("--noplot", action="store_true", help="Set this flag so that we don't generate any plots at the end of execution.")
    parser.add_argument("--retry", type=int, default=0, help="The number of times we need to retry the method. So each time we use a different random starting point so that we might converge on error. The default is 0 retries => 1 pass")
    parser.add_argument("--A", type=float, default=500, help="The row penalty")
    parser.add_argument("--B", type=float, default=500, help="The col penalty")
    parser.add_argument("--C", type=float, default=200, help="The global inhibition")
    parser.add_argument("--D", type=float, default=500, help="The distance penalty")
    parser.add_argument("--stopatitercnt", action='store_true', help="Setting this flag will not run the simulation beyond the value specified by itercnt")
    return parser.parse_args()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import _thread
    
    args = get_cmdline_args()

    N_CITY = args.ncity
    MAX_X = 20
    MAX_Y = 20

    delta_t = args.delta_t
    u0 = args.u0
    iteration_count = args.itercnt
    termination_threshold = args.termthresh
    non_interactive = args.nonint
    no_plot = args.noplot

    params.A = args.A
    params.B = args.B
    params.C = args.C
    params.D = args.D

    retries_left = args.retry
    
    stop_at_iteration_count = args.stopatitercnt

    # if we have passed it a file name then we better use that
    city, dist = None, None
    if args.file:
        city = pickle.load(open(args.file,"rb"))
        dist = []
        from math import sqrt as sqrt
        def dist_fx(a,b):
            return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        for i in range(len(city)):
            dist.append([])
            for j in range(len(city)):
                dist[i].append([])
                dist[i][j] = dist_fx(city[i],city[j])
    else:
        city, dist = rand_city(N_CITY, MAX_X, MAX_Y)
    print(city)
    # for row in dist:
    #    print (row)

    trc = tracer.TSPTracer(args.ncity, args.file, delta_t, u0, termination_threshold, params.A, params.B, params.C, params.D, city, dist)
    
    #_thread.start_new_thread(plot_cities, (city,))
    
    ax = None
    energy_axes = None
    
    if not no_plot:
        plt.ion()
        cities_plot = plt.figure()
        ax = cities_plot.add_subplot(111)
        ax.scatter(city[:,0], city[:,1])

        # This for the energy function:
        energy_plot = plt.figure()
        energy_axes = energy_plot.add_subplot(111)
        energy_axes.set_autoscale_on(True)


    # _thread.start_new_thread(hopfield_tsp,(city, dist))
    # import threading
    # hnnt = threading.Thread(target=hopfield_tsp, args=(city, dist))
    # hnnt.start()

    while retries_left >= 0:
        retries_left -= 1
        ret = hopfield_tsp(city,dist,energy_axes, iteration_count)
        if ret[0]:
	    # This implies a route was found.
            # We write the tracer contents
            trc.set_number_of_retries(args.retry - retries_left - 1)
            trc.set_status("pass")
            trc.save_to_stats_file()
            if not no_plot:
                import matplotlib.lines as lines
                edge_line_gen = lambda x,y : lines.Line2D([city[x][0], city[y][0]], 
                                                [city[x][1], city[y][1]],
                                                linewidth=1,
                                                linestyle=':')
                tsp = ret[2]
                edge_lines = [edge_line_gen(list(tsp[:,i]).index(1), list(tsp[:,(i+1)%len(city)]).index(1)) for i in range(len(city))]
                
                for line in edge_lines:
                    ax.add_line(line)
                input()
            exit(0);

    trc.set_status("fail")
    trc.set_number_of_retries(args.retry)
    trc.save_to_stats_file()
        # hnnt.join()
        # input()
