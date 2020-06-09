# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
import collections
import scipy.spatial
import time

# Custom libraries
from utils import GeoTools
from shapes import Point


class NotInGoalError(Exception):
    pass


class KDTree:
    """Wrapper class for scipy.spatial.cKDTree

    Args:
        data (np.array): List of 2D coordinates, of shape (?, 2).

    Attributes:
        tree (scipy.spatial.cKDTree): Internal kd-tree data structure.
    """

    def __init__(self, data):
        self.tree = scipy.spatial.cKDTree(data)

    def get_knn(self, query, k=1):
        """Return the distances and the indices
        of the k nearest neighbors to a query point.
        """
        return self.tree.query(query, k=k)


class Graph:
    """Graph class for 2D MPAP.

    Args:
        x_init (np.array): Initial state.
        x_range (np.array): Bounds of the state space, of shape (d, 2).
        env (Environment): Environment object defining obstacles.
        planner (LQRPlanner): (Already instantiated) planner for optimal control
            between two nodes.

    Attributes:
        x_init (np.array): Initial state.
        d (int): Dimension of the state space.
        x_range (np.array): Bounds of the state space, of shape (d, 2).
        env (Environment): Environment object defining obstacles.
        planner (LQRPlanner): Planner for optimal control between two nodes.
        samples (np.array): Sampled nodes of the graph, of shape (?, d)
        edges (defaultdict(defaultdict(tuple))): Edges of the graph, of the form
            {u:
                {v: [(cost(u,v), traj(uv))]}
            }
        skdtree (scipy.spatial.cKDTree): KDTree of the 2D coordinates of self.samples.
        r (float): Optimal cost threshold.
        lines_seen (list): list where the element of the ith idx is a dictionary containing
            the lines belonging to env that are seen by the ith vertex of the graph.
    """

    def __init__(self, x_init, x_range, environment, planner):
        self.x_init = x_init
        self.dim = len(x_init)
        self.init_samples()
        self.x_range = x_range
        self.env = environment
        self.planner = planner

    def initialize(self):
        self.init_samples()
        self.init_edges()
        self.lines_seen = []
        self.r = None
        self.update_kdtree()

    def init_samples(self):
        """Initializes the samples (nodes) and the edges.
        """
        self.samples = np.zeros((1, self.dim))
        self.samples[0, :] = self.x_init
        self.update_kdtree()

    def init_edges(self):
        """Initializes the edges.
        """
        self.edges = collections.defaultdict(lambda: collections.defaultdict(tuple))
        self.controls = collections.defaultdict(lambda: collections.defaultdict())
        self.Qs_scaled = collections.defaultdict(lambda: collections.defaultdict(float))

    def update_kdtree(self):
        """Update internal KDTree."""
        self.skdtree = KDTree(self.samples[:, :2])

    def clear(self):
        """Wrapper around self.init_samples().
        Meant to be called outside the class.
        """
        self.initialize()

    def set_samples(self, samples):
        self.samples = samples
        self.update_kdtree()

    def get_heuristic_r(self, n):
        eps = self.env.vol_free / n
        return eps * (1 + 3.5 * eps)

    def build(
        self,
        n=None,
        r=None,
        max_neighbors=6,
        config="worst",
        tol=1e-2,
        timing=True,
        motion_noise_ratio=0.05,
    ):
        """Build the graph.
        Args:
            n (int): Number of nodes to sample.
            r (float): Optimal cost threshold, i.e.
                (discard adding a node v to a node u if cost(u,v) > r).
            max_neighbors (int or None): Maximum number of neighbors to
                consider per node. If set to None, equivalent to no cap.
        """
        if r is None:
            r = self.get_heuristic_r(n)
        self.r = r
        not_sampled = len(self.samples) <= 1
        if not_sampled:
            self.init_samples()
            t0 = time.time()
            self.sample_free(n, config)
            t1 = time.time()
            if timing:
                print("Sampling took: {:0.2f} s.".format(t1 - t0))
        self.init_edges()
        t2 = time.time()
        if max_neighbors is None:
            max_neighbors = len(self.samples)
        self.connect(0, r, max_neighbors, config, tol, motion_noise_ratio)
        for src_idx in range(len(self.samples)):
            self.connect(src_idx, r, max_neighbors, config, tol, motion_noise_ratio)
        t3 = time.time()
        if timing:
            print("Connecting took: {:0.2f} s.".format(t3 - t2))

    def fromarray(self, array):
        return Point(array[0], array[1])

    def sample_free(self, n, config, goal_region=None, deterministic=True):
        """Sample n nodes in the free space."""
        if deterministic and self.dim == 2:
            new_samples = self.get_deterministic_samples(n, config, goal_region)
        else:
            self.num_added = 0
            new_samples = []
            while self.num_added < n:
                try:
                    sample = GeoTools.sample_in_range(self.x_range)
                    if self.num_added == 0 and goal_region is not None:
                        if not goal_region.contains(sample):
                            raise NotInGoalError
                    if not self.env.contains(self.fromarray(sample), config):
                        new_samples.append(sample)
                        self.num_added += 1
                except NotInGoalError:
                    pass
        self.add_samples(new_samples)
        print("Added {} samples.".format(self.num_added))
        self.update_kdtree()

    def get_deterministic_samples(
        self, n, config, goal_region=None, grid_noise_ratio=0.01
    ):
        """Deterministically sample nodes in the free space from a fixed size meshgrid.
        WARNING: Assumes 2D state space.
        """
        range_x = self.env.x_range[0][1] - self.env.x_range[0][0]
        range_y = self.env.x_range[1][1] - self.env.x_range[1][0]
        env_scale = range_x * range_y
        grid_res = env_scale / n
        x = np.arange(
            self.env.x_range[0][0] + 0.01 * range_x,
            self.env.x_range[0][1] - 0.01 * range_x,
            grid_res,
        )
        y = np.arange(
            self.env.x_range[1][0] + 0.01 * range_y,
            self.env.x_range[1][1] - 0.01 * range_y,
            grid_res,
        )
        xs, ys = np.meshgrid(x, y)
        self.num_added = 0
        new_samples = []
        for i in range(xs.shape[0]):
            for j in range(xs.shape[1]):
                sample = np.array([xs[i, j], ys[i, j]]) + np.random.multivariate_normal(
                    np.zeros(2), grid_noise_ratio * grid_res * np.eye(2)
                )
                if not self.env.contains(self.fromarray(sample), config):
                    new_samples.append(sample)
                    self.num_added += 1
        return new_samples

    def add_samples(self, new_samples):
        """Add a sample to the current set of samples."""
        self.samples = np.vstack((self.samples, new_samples))

    def connect(self, src_idx, r, max_neighbors, config, tol, motion_noise_ratio):
        """Connect src_idx'th node to any other node in the graph,
        in the limit of a total of max_neighbors, for which the optimal
        cost is less than r and for which the optimal trajectory lies in the free space.

        Args:
            src_idx (int): Index of the node for which we are searching for neighbors.
            r (float): Optimal cost threshold, i.e.
                (discard adding a node v to a node u if cost(u,v) > r).
            max_neighbors (int): Maximum number of neighbors to
                consider per node.
        """
        for dest_idx in self.skdtree.get_knn(
            self.samples[src_idx], k=min(max_neighbors + 1, len(self.samples))
        )[1][1:]:
            if dest_idx not in self.edges[src_idx]:
                cost, traj, controls = self.compute_path(
                    self.samples[src_idx], self.samples[dest_idx], tol
                )
                if cost <= r and not self.intersects(traj, config):
                    self.add_edge(
                        src_idx, dest_idx, cost, traj, controls, motion_noise_ratio
                    )
                    # Add the edge the other way if not already
                    if src_idx not in self.edges[dest_idx]:
                        cost_rev, traj_rev, controls_rev = self.compute_path(
                            self.samples[dest_idx], self.samples[src_idx], tol
                        )
                        self.add_edge(
                            dest_idx,
                            src_idx,
                            cost_rev,
                            traj_rev,
                            controls_rev,
                            motion_noise_ratio,
                        )

    def add_edge(self, src_idx, dest_idx, cost, traj, controls, motion_noise_ratio):
        self.edges[src_idx][dest_idx] = (cost, traj)
        self.controls[src_idx][dest_idx] = controls
        controls_norms = np.linalg.norm(controls, axis=1)
        motion_noise_max = np.diag(np.abs(traj[1] - traj[0]) * motion_noise_ratio)
        if np.any(np.max(controls_norms)) < 1e-16:
            print("src_idx = {}".format(src_idx))
            print("dest_idx = {}".format(dest_idx))
            print("controls_norms = {}".format(controls_norms))
            print("np.max(controls_norms) = {}".format(np.max(controls_norms)))
        control_scales = np.concatenate(
            (np.zeros(1), controls_norms / np.max(controls_norms))
        )
        self.Qs_scaled[src_idx][dest_idx] = [
            control_scale * motion_noise_max for control_scale in control_scales
        ]

    def compute_path(self, u, v, tol):
        """Wrapper around the planner attribute."""
        return self.planner.compute_path(u, v, tol=tol)

    def intersects(self, traj, config):
        """Return false if a trajectory lies in the free space."""
        for point_idx in range(len(traj) - 1):
            p = self.fromarray(traj[point_idx])
            p_ = self.fromarray(traj[point_idx + 1])
            if self.env.is_intersected([p, p_], config):
                return True
        return False

    def update_lines_seen(self):
        """Update the lines seen by each vertex of the graph."""
        for sample_idx in range(len(self.samples)):
            self.lines_seen.append(
                self.env.get_lines_seen(self.fromarray(self.samples[sample_idx]))
            )

    def indices2path(self, indices_list):
        path = []
        for idx in indices_list:
            path.append(self.samples[idx, :2])
        return path

    def plot(
        self,
        ax=None,
        plot_edges=True,
        show_idx=False,
        show_id=True,
        show_edges_id=True,
        show_full=True,
        show_actual_errors=False,
    ):
        """Plot the the nodes and edges of the graph.
        Note: this can take a long time if plot_edges is set to True.
        """
        self.env.plot(
            ax=ax,
            show_id=show_id,
            show_edges_id=show_edges_id,
            show_full=show_full,
            show_actual_errors=show_actual_errors,
        )
        x_init_x, x_init_y = self.x_init[:2]
        plt.scatter(x_init_x, x_init_y, color="r", label="x_init")
        if show_idx:
            plt.annotate(
                str(0), [x_init_x, x_init_y], fontsize=10,
            )
            for i in range(len(self.samples[1:,])):
                sample_x, sample_y = self.samples[1 + i, :2]
                plt.scatter(sample_x, sample_y, color="g")
                plt.annotate(
                    str(i + 1), [sample_x, sample_y], fontsize=10,
                )
        else:
            plt.scatter(
                self.samples[1:, 0], self.samples[1:, 1], color="g", s=30
            )
        if plot_edges:
            self.plot_edges()

    def plot_edges(self):
        """Plot the edges of the graph."""
        been_there = set()
        for src_idx in self.edges:
            for dest_idx in self.edges[src_idx]:
                if (src_idx, dest_idx) and (dest_idx, src_idx) not in been_there:
                    plt.plot(
                        self.edges[src_idx][dest_idx][1][:, 0],
                        self.edges[src_idx][dest_idx][1][:, 1],
                        linestyle="--",
                        linewidth=0.5,
                        color="k",
                        alpha=1.0,
                    )
                    been_there.add((src_idx, dest_idx))
                    been_there.add((dest_idx, src_idx))
