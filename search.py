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


# def process_plan(
#     samples, edges, scales, env, prob_threshold, scaling_factors, p, P
# ):
#     print(
#         "========== p = {} at index {} ==========".format(
#             samples[p.head_idx], p.head_idx
#         )
#     )
#     print("NEIGHBORS: {}".format(edges[p.head_idx].keys()))
#     print("p.path_indices = {}".format(p.path_indices))

#     start = time.time()

#     plans_to_add = []

#     for neighbor_idx, v in edges[p.head_idx].items():
#         if neighbor_idx in set(p.path_indices):
#             continue
#         dominated = False
#         discard = False
#         to_neighbor_cost, to_neighbor_path = v
#         print("----- neighbor_idx = {} -----".format(neighbor_idx))
#         p_copy = deepcopy(p)
#         for i, sub_neighbor in enumerate(to_neighbor_path[1:]):
#             p_copy.add_point(
#                 env, sub_neighbor, scales[p.head_idx][neighbor_idx][i],
#             )
#             prob_collision = p_copy.get_max_prob_collision(env, scaling_factors)
#             if prob_collision >= prob_threshold:
#                 discard = True
#                 break
#         if discard:
#             continue
#         p_copy.update_info(neighbor_idx, to_neighbor_cost, to_neighbor_path)

#         if neighbor_idx in P:
#             for q in P[neighbor_idx]:
#                 lower_cost = q.cost < p_copy.cost
#                 enclosed = q.Xk_full <= p_copy.Xk_full
#                 if lower_cost and enclosed:
#                     dominated = True
#                     break

#         if not dominated:
#             plans_to_add.append((p_copy, neighbor_idx))

#     elapsed = time.time() - start
#     print(
#         "Done p = {} at index {}, took {} seconds".format(
#             samples[p.head_idx], p.head_idx, elapsed
#         )
#     )

#     return plans_to_add


def process_plan(samples, edges, scales, env, prob_threshold, scaling_factors, p):
    print(
        "========== p = {} at index {} ==========".format(
            samples[p.head_idx], p.head_idx
        )
    )
    print("NEIGHBORS: {}".format(edges[p.head_idx].keys()))
    print("p.path_indices = {}".format(p.path_indices))

    start = time.time()

    plans_to_add = []

    for neighbor_idx, v in edges[p.head_idx].items():
        if neighbor_idx in set(p.path_indices):
            continue
        discard = False
        to_neighbor_cost, to_neighbor_path = v
        print("----- neighbor_idx = {} -----".format(neighbor_idx))
        p_copy = deepcopy(p)
        for i, sub_neighbor in enumerate(to_neighbor_path[1:]):
            p_copy.add_point(
                env, sub_neighbor, scales[p.head_idx][neighbor_idx][i],
            )
            prob_collision = p_copy.get_max_prob_collision(env, scaling_factors)
            if prob_collision >= prob_threshold:
                discard = True
                break
        if discard:
            continue
        p_copy.update_info(neighbor_idx, to_neighbor_cost, to_neighbor_path)
        plans_to_add.append((p_copy, neighbor_idx))

    elapsed = time.time() - start
    print(
        "Done p = {} at index {}, took {} seconds".format(
            samples[p.head_idx], p.head_idx, elapsed
        )
    )

    return plans_to_add


def process_dominated(P_head_idx, p_head_idx, p_cost, p_Xk_full, plan_number):
    if len(P_head_idx) > 0:
        to_remove_from_P_open = set()
        for q in P_head_idx:
            lower_cost = q.cost < p_cost
            enclosed = q.Xk_full <= p_Xk_full
            if lower_cost and enclosed:
                to_remove_from_P_open.add(plan_number)
        return p_head_idx, to_remove_from_P_open


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
        self.P_open_dict[self.plan_number] = p_init
        self.P[0] = set()
        self.P[0].add(deepcopy(p_init))
        self.G = self.P_open.copy()

    def clear(self):
        self.P_open = set()
        self.P_open_dict = {}
        self.P = collections.defaultdict(set)
        self.G = set()

    def set_goal(self, goal_region):
        """Input must be a Rectangle object.
        """
        self.goal_region_rec = goal_region
        self.goal_region = goal_region.as_poly["original"]
        self.store_ratios()

    def store_ratios(self):
        self.ratios = np.zeros(len(self.graph.samples))
        for idx, sample in enumerate(self.graph.samples[:, :2]):
            self.ratios[idx] = self.get_dist_ratio(sample)

    def get_dist_ratio(self, sample):
        dist2origin = np.linalg.norm(sample - self.x_init[:2])
        if np.abs(dist2origin) < 1e-16:
            return 0
        dist2goal = self.goal_region_rec.get_min_dist(Point(sample[0], sample[1]))[0]
        return (dist2origin + dist2goal) / dist2origin

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

    def remove_dominated_parallel(self, n_jobs, prune=True):
        num_open_plans = len(self.P_open)
        to_remove_from_P_open = set()
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_dominated)(
                self.P[p.head_idx], p.head_idx, p.cost, p.Xk_full, plan_number
            )
            for p, plan_number in self.P_open
        )
        for head_idx, to_remove_from_P_open in results:
            for plan_number in to_remove_from_P_open:
                p_to_remove = self.P_open_dict[plan_number]
                self.P_open.remove((p_to_remove, plan_number))
                self.P[head_idx].remove(p_to_remove)
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
                key=lambda p_tuple: p_tuple[0].cost * self.ratios[p_tuple[0].head_idx],
            )
            num_open_plans_to_remove = int(portion * num_open_plans) + 1
            self.P_open = set(P_open_sorted[:-num_open_plans_to_remove])
            print("Pruned {} plans in P_open.".format(num_open_plans_to_remove))

    def prune(self, i):
        if self.pruning_coeff == 1:
            self.G = self.P_open.copy()
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
        remove_dominated_parallel=True,
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
        time.sleep(2)  # sleep for 2 s so that I can read it

        i = 0  # counter for iteration
        while self.P_open and not self.reached_goal(early_termination):
            print("Processing %d plans" % len(self.G))
            # results = Parallel(n_jobs=n_jobs)(
            #     delayed(process_plan)(
            #         self.graph.samples,
            #         self.graph.edges,
            #         self.graph.scales,
            #         self.graph.env,
            #         self.prob_threshold,
            #         scaling_factors,
            #         p,
            #         {
            #             key: self.P[key]
            #             for key in self.P
            #             if key in self.graph.edges[p.head_idx]
            #         },
            #     )
            #     for p, _ in self.G
            # )
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_plan)(
                    self.graph.samples,
                    self.graph.edges,
                    self.graph.scales,
                    self.graph.env,
                    self.prob_threshold,
                    scaling_factors,
                    p,
                )
                for p, _ in self.G
            )
            for result in results:
                for plan, neighbor_idx in result:
                    self.plan_number += 1
                    self.P[neighbor_idx].add(plan)
                    self.P_open.add((plan, self.plan_number))
                    self.P_open_dict[self.plan_number] = plan

            print(80 * "=")
            print("Done iteration %d" % i)
            print(80 * "=")
            t1 = time.time()
            if remove_dominated_parallel:
                self.remove_dominated_parallel(n_jobs=n_jobs)
            else:
                self.remove_dominated()
            print("remove_dominated took {} s.".format(time.time() - t1))
            self.P_open -= self.G
            i += 1
            self.prune(i)

        P_candidates = self.get_candidates()
        print("P_candidates = {}".format(P_candidates))
        if not P_candidates:
            print("Could not find a path.")
            return None
        return P_candidates, min(P_candidates, key=lambda plan: plan.cost)
