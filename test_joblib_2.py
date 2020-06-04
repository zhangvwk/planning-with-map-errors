from shapes import Rectangle
from environment import Environment2D
from graph import Graph
from lqr import LQRPlanner
from search import Searcher
from plan import PlanUtils
import numpy as np
import json


x_lims = [-40, 15]
y_lims = [-5, 55]
env = Environment2D(x_lims, y_lims)
rec_0 = Rectangle(0, -35, 0, 25, 50, 0)
rec_0.set_error_bounds(np.array([0, -5, 0, 0]), np.array([0, 5, 0, 0]))
# rec_0.set_actual_errors(np.array([0,-4,0,0]))
# rec_0.set_actual_errors(np.array([0,4,0,0]))
env.add_rectangles([rec_0])
goal_region = Rectangle(-1, 0, 45, 5, 5, 0)
dt = 0.2
A = np.eye(2)
B = dt * np.eye(2)
Q_lqr = np.eye(2)
R_lqr = np.eye(2)
lqr_planner = LQRPlanner(Q_lqr, R_lqr, A, B)
Q = np.eye(2) * 0.05
R = 0.1
T = 50
x0 = [0, 0]
x_est_0 = x0
P_est_0 = np.eye(2)
x_range = np.array([x_lims, y_lims])
nsamples = 50
tol = 1e-2
g = Graph(x0, x_range, env, lqr_planner)
g.clear()
samples_all = np.load("samples_300.dat", allow_pickle=True)

# samples_manual = []
# for i in [0, 22, 8, 17, 18, 23, 13, 47, 10, 29, 11]:
#     samples_manual.append(samples_all[i, :2])
# samples_manual = np.array(samples_manual)
# g.set_samples(samples_manual)
g.set_samples(samples_all)
g.build(nsamples, r=10, max_neighbors=5, tol=tol)
# fig, ax = plt.subplots()
# g.plot(ax=ax, show_idx=True)
# goal_region.plot(ax=ax,as_goal=True)
# goal_region.as_poly["original"].plot(ax=ax)
# # PlotTools.plot_traj(g.indices2path(plan_found.path_indices))
# plt.show()
horizon = 60
kmax = PlanUtils.get_kmax(dt, horizon)
lambda_coeff = 1.0
searcher = Searcher(g, lambda_coeff * g.r)
searcher.set_source()
searcher.initialize_open(Q, R, P_est_0, kmax)
searcher.set_goal(goal_region)
collision_thresh = 0.2
P_candidates, plan_found = searcher.explore(collision_thresh)
print("plan_found.path_indices = {}".format(plan_found.path_indices))
with open("path_indices_300.txt", "w") as outfile:
    json.dump(plan_found.path_indices, outfile)
