# Standard libraries
import numpy as np
import time

# Custom libraries
from shapes import Rectangle
from environment import Environment2D
from graph import Graph
from lqr import LQRPlanner
from search import Searcher
from plan import PlanUtils
from utils import ProcTools
import matplotlib.pyplot as plt

""" Environment """
x_lims = [-10, 10]
y_lims = [-12, 10]
env = Environment2D(x_lims, y_lims)

rec_0 = Rectangle(0, -8, 7, 7, 2, 0)
rec_1 = Rectangle(1, 3, 7, 4, 2, 0)
rec_2 = Rectangle(2, -4, 2, 9, 1.5, 0)
rec_3 = Rectangle(3, -3, -2.5, 2, 2, 0)
rec_4 = Rectangle(4, -8, -11, 3, 3, 0)
rec_5 = Rectangle(5, 2, -10.5, 2.5, 2.5, 0)
rec_6 = Rectangle(6, 3.5, -3, 2, 2, 45)

rec_0.set_error_bounds(np.array([-0.5, 0, 0, 0]), np.array([1.5, 0, 0.5, 0]))
rec_1.set_error_bounds(np.array([0, 0, 0, 0]), np.array([1, 1.5, 0, 0.5]))
rec_2.set_error_bounds(np.array([0, 0, 0, 0]), np.array([0.5, 0, 0.5, 0.5]))
rec_3.set_error_bounds(np.array([0, 0, 0, 0]), np.array([0, 0, 1, 0]))
rec_4.set_error_bounds(np.array([0, -0.5, 0, 0]), np.array([0, 2, 1.5, 0]))
rec_5.set_error_bounds(np.array([0, 0, 0, 0]), np.array([0, 0, 0.5, 1.5]))
rec_6.set_error_bounds(np.array([0, 0, 0, 0]), np.array([0, 0, 1.0, 0.5]))

env.add_rectangles([rec_0, rec_1, rec_2, rec_3, rec_4, rec_5, rec_6])
goal_region = Rectangle(-1, 6.5, 2, 2, 2, 0)

""" Planner """
dt = 0.2
A = np.eye(2)
B = dt * np.eye(2)
Q_lqr = np.eye(2)
R_lqr = 0.1 * np.eye(2)
lqr_planner = LQRPlanner(Q_lqr, R_lqr, A, B)

""" Graph """
x0 = [-8, 2]
x_range = np.array([x_lims, y_lims])
tol = 1e-2
g = Graph(x0, x_range, env, lqr_planner)
g.clear()
g.set_samples(np.load("samples/samples_env2_tilted.dat", allow_pickle=True))
g.build(r=2, max_neighbors=3, tol=tol, motion_noise_ratio=0.05)

print(g.edges[101])

""" Initial estimates """
# We are confident about our initial estimate up to 0.5% of the environment size.
init_confidence = 0.5 / 100

x_est_0 = x0
P_est_0 = init_confidence * np.diag([(x_lims[1] - x_lims[0]), y_lims[1] - y_lims[0]])

""" Searcher """
horizon = 30  # After 30s, we assume the beginning is forgotten.
kmax = PlanUtils.get_kmax(dt, horizon)
lambda_coeff = 1.0
R = 0.1
searcher = Searcher(g, lambda_coeff * g.r)
searcher.set_source()
searcher.initialize_open(R, P_est_0, kmax)
searcher.set_goal(goal_region)

""" Explore """
prob_threshold = 0.01  # Safe = less than 1% chance of collision.
prune_portion = 0.1
t1 = time.time()
P_candidates, plan_found = searcher.explore(
    prob_threshold=prob_threshold,
    prune_portion=prune_portion,
    remove_dominated_parallel=False,
)
print("Terminated in {} s.".format(time.time() - t1))

ProcTools.dump_plans(P_candidates, "samples/candidates_env2.txt")
