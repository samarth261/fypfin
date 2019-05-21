#!/usr/bin/python3.5

# This file defines a way where we can save some of the states into an output file while generating the results for the tsp

# COLS:
# unique_id(time), pickle_file_name, city_file_name,status(pass/fail), n, delta_t, u0, termthresh, A, B, C, D, soln_dist, number of retries

STAT_FILE_NAME = "stats.csv"

import time
import uuid
import pickle

class TSPTracer:
    def __init__(self, n_cities, city_file, delta_t, u0, termthresh, A, B, C, D, cities, dists):
        # We just store all of the here nothing more
        self.n = len(cities)
        self.delta_t = delta_t
        self.u0 = u0
        self.termination_threshold = termthresh
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.cities = cities
        self.dists = dists
        self.tsp_soln = None
        self.soln_dist = None

        self.true_iter = None
        self.init_net = None

        self.number_of_retries = None
        self.status = None # pass or fail

        self.unique_id = uuid.uuid4()
        self.pickle_file_name = "tracers/" + str(self.unique_id) # Generating a random file name
        self.city_file = None

        if city_file:
            # implies a file was given
            self.city_file = city_file
        else:
            # it's a new random graph
            # so we need to pickle that too
            self.city_file = "instances/" + self.new_fname()
            pickle.dump(self.cities, open(self.city_file, "wb"))

    def new_fname(self):
        fname = str(self.n)
        cnt = int(open("prob_num.txt").read())
        cnt += 1
        open("prob_num.txt","w").write(str(cnt))
        fname += "_" + str(cnt) + "_" + str(self.unique_id)
        return fname

    def add_solution(self, soln):
        # soln needs to be a 0 or 1 matrix which is correct
        self.soln = soln
        
        total_distance = 0
        for x in range(self.n):
            for y in range(self.n):
                if y == x:
                    continue
                for i in range(self.n):
                    total_distance += ( self.dists[x][y] * self.soln[x][i] * 
                                    (self.soln[y][(i+1)%self.n] + self.soln[y][(i-1)%self.n]) )
        self.soln_dist = total_distance

    def set_true_itercnt_and_init_network(self, true_iter, init_net):
        self.true_iter = true_iter
        self.init_net = init_net

    def set_number_of_retries(self, retry_cnt):
        # print("\n",retry_cnt)
        # input()
        self.number_of_retries = retry_cnt

    def set_status(self, pass_fail):
        self.status = pass_fail

    def save_to_stats_file(self):
        # First choose the appropriate stats file, also we pickle this entire object and store it
        stat_str = ",".join(map(str, [str(time.time()),
                                      self.pickle_file_name,
                                      self.city_file,
                                      self.status,
                                      self.n,
                                      self.delta_t,
                                      self.u0,
                                      self.termination_threshold,
                                      self.A,
                                      self.B,
                                      self.C,
                                      self.D,
                                      self.true_iter,
                                      self.soln_dist,
                                      self.number_of_retries]))
        stats = open(STAT_FILE_NAME, "a")
        stats.write(stat_str+"\n")
        stats.close()

        pickle.dump(self, open(self.pickle_file_name, "wb"))
