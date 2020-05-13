# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
import collections
import scipy.spatial

# Custom libraries
from utils import GeoTools


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
    """

    def __init__(self, x_init, x_range, environment, planner):
        self.x_init = x_init
        self.d = len(x_init)
        self.init_samples()
        self.x_range = x_range
        self.env = environment
        self.planner = planner

    def init_samples(self):
        """Initializes the samples (nodes) and the edges.
        """
        self.samples = np.zeros((1, self.d))
        self.samples[0, :] = self.x_init
        self.edges = collections.defaultdict(lambda: collections.defaultdict(tuple))
        self.update_kdtree()

    def update_kdtree(self):
        """Update internal KDTree."""
        self.skdtree = KDTree(self.samples[:, :2])

    def clear(self):
        """Wrapper around self.init_samples().
        Meant to be called outside the class.
        """
        self.init_samples()

    def build(self, n, r, max_neighbors=10):
        """Build the graph.
        Args:
            n (int): Number of nodes to sample.
            r (float): Optimal cost threshold, i.e.
                (discard adding a node v to a node u if cost(u,v) > r).
            max_neighbors (int or None): Maximum number of neighbors to
                consider per node. If set to None, equivalent to no cap.
        """
        self.sample_free(n)
        if max_neighbors is None:
            max_neighbors = len(self.samples)
        for src_idx in range(len(self.samples)):
            self.connect(src_idx, r, max_neighbors)

    def sample_free(self, n):
        """Sample n nodes in the free space."""
        num_added = 0
        new_samples = []
        while num_added < n:
            sample = self.sample_in_range()
            if not self.env.contains(GeoTools.array2point(sample)):
                new_samples.append(sample)
                num_added += 1
        self.add_sample(new_samples)
        self.update_kdtree()

    def sample_in_range(self):
        """Sample a node uniformly in the state space."""
        return (self.x_range[:, 1] - self.x_range[:, 0]) * np.random.uniform(
            size=self.d
        ) + self.x_range[:, 0]

    def add_sample(self, new_samples):
        """Add a sample to the current set of samples."""
        self.samples = np.vstack((self.samples, new_samples))

    def connect(self, src_idx, r, max_neighbors):
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
            self.samples[src_idx], k=min(max_neighbors, len(self.samples))
        )[1][1:]:
            cost, traj = self.compute_path(
                self.samples[src_idx], self.samples[dest_idx]
            )
            if cost <= r and not self.intersects(traj):
                self.edges[src_idx][dest_idx] = (cost, traj)

    def compute_path(self, u, v):
        """Wrapper around the planner attribute."""
        return self.planner.compute_path(u, v)

    def intersects(self, traj):
        """Return false if a trajectory lies in the free space."""
        for point_idx in range(len(traj) - 1):
            p = GeoTools.array2point(traj[point_idx])
            p_ = GeoTools.array2point(traj[point_idx + 1])
            if self.env.is_intersected([p, p_]):
                return True
        return False

    def plot(self, plot_edges=True):
        """Plot the the nodes and edges of the graph.
        Note: this can take a long time if plot_edges is set to True.
        """
        self.env.plot()
        plt.scatter(self.x_init[0], self.x_init[1], color="r", label="x_init")
        plt.scatter(self.samples[1:, 0], self.samples[1:, 1])
        if plot_edges:
            self.plot_edges()

    def plot_edges(self):
        """Plot the edges of the graph."""
        for src_idx in self.edges:
            for dest_idx in self.edges[src_idx]:
                plt.plot(
                    self.edges[src_idx][dest_idx][1][:, 0],
                    self.edges[src_idx][dest_idx][1][:, 1],
                    linestyle="--",
                    color="k",
                    alpha=0.25,
                )
