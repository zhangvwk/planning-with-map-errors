# Standard libraries
import numpy as np
import collections

# Custom libraries
from plan import Plan
from utils import GeoTools


class Searcher:
    def __init__(self, graph, pruning_coeff=0.5):
        self.graph = graph
        self.pruning_coeff = pruning_coeff
        self.x_init = None
        self.goal_region = None
        self.P_open = set()
        self.P = collections.defaultdict(set)
        self.G = set()
        self.prob_threshold = 1.0

    def set_source(self, x_init=None):
        if not x_init:
            self.x_init = self.graph.samples[0, :]
        else:
            self.x_init = x_init
        self.initialize_open()

    def initialize_open(self):
        self.P_open.add(Plan(self.x_init, None, 0, None))
        self.G = self.P_open

    def set_goal(self, goal_region):
        self.goal_region = goal_region

    def is_valid_goal(self, verbose=True):
        return GeoTools.is_valid_region(self.goal_region, self.graph.env, verbose)

    def reached_goal(self):
        return len([p for p in self.G if self.in_goal(p[0])]) != 0

    def in_goal(self, point):
        return self.goal_region.A.dot(point) <= self.goal.b

    def remove_dominated(self):
        for p in self.P_open:
            for q in self.P:
                if q.head != p.head:
                    continue
                else:
                    lower_cost = q.cost < p.cost
                    enclosed = q.zonotope_state <= p.zonotope_state
                    if lower_cost and enclosed:
                        self.P_open.remove(p)
                        self.P.remove(p)

    def prune(self, i):
        for p in self.P_open:
            if p.cost <= i * self.pruning_coeff:
                self.G.add(p)

    def collision(self, prob_collision):
        return prob_collision <= self.prob_threshold

    def explore(self, prob_threshold):
        if self.x_init is None:
            print("Setting the source node as the first node in the graph.")
            self.set_source()
        try:
            assert self.is_valid_goal()
        except AssertionError:
            print("Goal region not defined.")
            raise
        self.prob_threshold = prob_threshold
        i = 0
        while not self.P_open and not self.reached_goal():
            for p in self.G:
                for k, v in self.graph.edges[p.head_idx].items():
                    discard = False
                    neighbor_idx = k
                    to_neighbor_cost, to_neighbor_path = v
                    for sub_neighbor in to_neighbor_path:
                        p.add_point(self.graph_env, sub_neighbor)
                        prob_collision = p.get_prob_collision(self.graph.env)
                        if self.collision(prob_collision):
                            break
                            discard = True
                    if discard:
                        continue
                    p.update_info(to_neighbor_cost, to_neighbor_path)
                    self.P[neighbor_idx].add(p)
                    self.P_open.add(p)
            self.remove_dominated()
            self.P_open -= self.G
            i += 1
            self.prune(i)
        P_candidates = self.get_candidates()
        return min(P_candidates, key=lambda plan: plan.cost)
