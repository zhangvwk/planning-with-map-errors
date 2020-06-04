# Standard libraries
import numpy as np
import collections
import multiprocessing
import time
from copy import deepcopy
from joblib import Parallel, delayed

# Custom libraries
from plan import Plan
from utils import GeoTools
from shapes import Point


def process_plan(searcher, scaling_factors, p):
    print(
        "========== p = {} at index {} ==========".format(
            searcher.graph.samples[p.head_idx], p.head_idx
        )
    )
    print("NEIGHBORS: {}".format(searcher.graph.edges[p.head_idx].keys()))
    print("p.path_indices = {}".format(p.path_indices))

    plans_to_add = []

    for neighbor_idx, v in searcher.graph.edges[p.head_idx].items():
        if neighbor_idx in set(p.path_indices):
            continue
        discard = False
        to_neighbor_cost, to_neighbor_path = v
        print("----- neighbor_idx = {} -----".format(neighbor_idx))
        p_copy = deepcopy(p)
        for i, sub_neighbor in enumerate(to_neighbor_path[1:]):
            p_copy.add_point(
                searcher.graph.env,
                sub_neighbor,
                searcher.graph.scales[p.head_idx][neighbor_idx][i],
            )
            prob_collision = p_copy.get_max_prob_collision(
                searcher.graph.env, scaling_factors
            )
            if searcher.collision(prob_collision):
                discard = True
                break
        if discard:
            continue
        p_copy.update_info(neighbor_idx, to_neighbor_cost, to_neighbor_path)
        plans_to_add.append((p_copy, neighbor_idx))

    return plans_to_add


class Searcher:
    def __init__(self, graph, pruning_coeff=0.5):
        self.graph = graph
        self.pruning_coeff = pruning_coeff
        self.x_init = None
        self.start_idx = None
        self.goal_region = None
        self.clear()
        self.prob_threshold = 1.0

    def set_source(self, start_idx=0):
        self.start_idx = start_idx
        self.x_init = self.graph.samples[start_idx, :]

    def initialize_open(self, Q, R, P0, kmax):
        self.clear()
        self.plan_number = 0
        p_init = Plan(
            self.x_init, self.start_idx, self.graph.env, self.x_init.shape[0], R, kmax,
        )
        # Set variables in p_init
        p_init.set_motion(self.graph.planner.A, self.graph.planner.B, Q)
        p_init.set_gain(self.graph.planner.gain)
        p_init.set_init_est(P0)
        self.P_open.add((p_init, self.plan_number))
        self.P[0] = set()
        self.P[0].add(deepcopy(p_init))
        self.G = self.P_open.copy()

    def clear(self):
        self.P_open = set()
        self.P = collections.defaultdict(set)
        self.G = set()

    def set_goal(self, goal_region):
        """Input must be a Rectangle object.
        """
        self.goal_region_rec = goal_region
        self.goal_region = goal_region.as_poly["original"]

    def is_valid_goal(self, config="worst", verbose=True):
        try:
            return GeoTools.is_valid_region(
                self.goal_region, self.graph.env, verbose, config=config
            )
        except AttributeError:
            print("Please set the goal region using .set_goal(goal_region).")

    def reached_goal(self, early_termination):
        if not early_termination:
            return False
        return len(
            [
                p
                for p, plan_number in self.G
                if self.in_goal(Point(p.head[0], p.head[1]))
            ]
        )

    def in_goal(self, point):
        if self.goal_region:
            return np.all(
                self.goal_region.A.dot(point.as_array()) <= self.goal_region.b
            )
        return False

    def remove_dominated(self, prune=True):
        num_open_plans = len(self.P_open)
        to_remove_from_P_open = set()
        for p, plan_number in self.P_open:
            if p.head_idx in self.P:
                to_remove_from_P = set()
                for q in self.P[p.head_idx]:
                    lower_cost = q.cost < p.cost
                    enclosed = q.Xk_full <= p.Xk_full
                    if lower_cost and enclosed:
                        to_remove_from_P_open.add((p, plan_number))
                        to_remove_from_P.add(p)
                self.P[p.head_idx] -= to_remove_from_P
        self.P_open -= to_remove_from_P_open
        print(
            "P_open went from {} to {} plans.".format(num_open_plans, len(self.P_open))
        )
        if prune:
            self.prune_alternate()

    def prune_alternate(self, portion=0.1):
        num_open_plans = len(self.P_open)
        if num_open_plans >= 10:
            print("num_open_plans = {}".format(num_open_plans))
            P_open_sorted = sorted(
                self.P_open,
                key=lambda p_tuple: p_tuple[0].cost
                * self.get_dist_ratio(p_tuple[0].head_idx),
            )
            num_open_plans_to_remove = int(portion * num_open_plans) + 1
            self.P_open = set(P_open_sorted[:-num_open_plans_to_remove])
            print("Pruned {} plans in P_open.".format(num_open_plans_to_remove))

    def get_dist_ratio(self, head_idx):
        head_location = self.graph.samples[head_idx, :2]
        dist2origin = np.linalg.norm(head_location - self.x_init[:2])
        dist2goal = self.goal_region_rec.get_min_dist(
            Point(head_location[0], head_location[1])
        )[0]
        ratio = (dist2origin + dist2goal) / dist2origin
        return ratio

    def prune(self, i):
        self.G = set()
        for p, plan_number in self.P_open:
            if p.cost <= i * self.pruning_coeff:
                self.G.add((p, plan_number))

    def collision(self, prob_collision):
        return prob_collision >= self.prob_threshold

    def get_candidates(self):
        P_candidates = set()
        for head_idx in self.P:
            if self.in_goal(
                Point(self.graph.samples[head_idx][0], self.graph.samples[head_idx][1])
            ):
                P_candidates.update(self.P[head_idx])
        return P_candidates

    def explore(
        self,
        prob_threshold=0.1,
        scaling_factors=[6, 5, 4, 3, 2, 1],
        early_termination=True,
    ):
        if self.x_init is None:
            print("Please set the source node using .set_source(x_init)")
            raise
        if len(self.P_open) == 0:
            print("Please initialize the searcher using .initialize_open(R, kmax).")
            raise
        try:
            assert self.is_valid_goal()
        except AssertionError:
            print("Invalid goal region.")
            raise
        self.prob_threshold = prob_threshold

        # count the number of available cores
        n_cpu = multiprocessing.cpu_count()
        n_jobs = max(n_cpu - 2, 1)
        print("I have detected %d cores, I am going to use %d" % (n_cpu, n_jobs))
        time.sleep(5)  # sleep for 5 s so that I can read it

        i = 0  # counter for iteration
        while self.P_open and not self.reached_goal(early_termination):
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_plan)(self, scaling_factors, p) for p, _ in self.G
            )
            results = [pairs for x in results for pairs in x]
            for plan, neighbor_idx in results:
                self.plan_number += 1
                self.P[neighbor_idx].add(plan)
                self.P_open.add((plan, self.plan_number))

            print(80 * "=")
            print("Done iteration %d" % i)
            print(80 * "=")
            self.remove_dominated()
            self.P_open -= self.G
            i += 1
            self.prune(i)

        P_candidates = self.get_candidates()
        print("P_candidates = {}".format(P_candidates))
        if not P_candidates:
            print("Could not find a path.")
            return None
        return P_candidates, min(P_candidates, key=lambda plan: plan.cost)
