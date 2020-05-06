# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
import collections

# Custom libraries
from shapes import Point


class LQRPlanner:
    # def __init__(self, Q, R, A, B):
    #     self.Q = Q
    #     self.R = R
    #     self.A = A
    #     self.B = B

    def __init__(self):
        pass

    def compute_path(self, u, v):
        cost = np.linalg.norm(u - v)
        path = np.vstack((u, v))
        return cost, path


class Graph:
    """ Graph class.
    Instantiated with an initial vertex, a state space range
    and an Environment object.
    """

    def __init__(self, x_init, x_range, environment):
        self.x_init = x_init
        self.d = len(x_init)
        self.init_samples()
        self.x_range = x_range
        self.env = environment
        # self.planner = LQRPlanner(Q, R, A, B)
        self.planner = LQRPlanner()

    def init_samples(self):
        self.samples = np.zeros((1, self.d))
        self.samples[0, :] = self.x_init
        self.edges = collections.defaultdict(lambda: collections.defaultdict(tuple))

    def clear(self):
        self.init_samples()

    def build(self, n, r):
        """Build the graph."""
        self.sample_free(n)
        for src_idx in range(len(self.samples)):
            self.connect(src_idx, r)

    def sample_free(self, n):
        """Sample n nodes in the free space."""
        num_added = 0
        while num_added < n:
            sample = self.sample_in_range()
            if not self.env.contains(Point(sample[0], sample[1])):
                self.add_sample(sample)
                num_added += 1

    def sample_in_range(self):
        """Sample a node uniformly in the state space."""
        return (self.x_range[:, 1] - self.x_range[:, 0]) * np.random.uniform(
            size=self.d
        ) + self.x_range[:, 0]

    def add_sample(self, sample):
        """Add a sample to the current set of samples."""
        self.samples = np.vstack((self.samples, sample))

    def connect(self, src_idx, r):
        """Connect src_idx'th node to any other node in the graph
        for which the optimal cost is less than r and for which
        the optimal trajectory lies in the free space.
        """
        for dest_idx in range(len(self.samples)):
            if dest_idx != src_idx and dest_idx not in self.edges[src_idx]:
                cost, traj = self.compute_path(
                    self.samples[src_idx], self.samples[dest_idx]
                )
                if cost <= r and not self.intersects(traj):
                    self.edges[src_idx][dest_idx] = (cost, traj)

    def compute_path(self, u, v):
        """Wrapper around the planner attribute."""
        return self.planner.compute_path(u, v)

    def intersects(self, traj):
        """Return true if a trajectory lies in the free space."""
        for point_idx in range(len(traj) - 1):
            p = Point(traj[point_idx][0], traj[point_idx][1])
            p_ = Point(traj[point_idx + 1][0], traj[point_idx + 1][1])
            if self.env.is_intersected([p, p_]):
                return True
        return False

    def plot(self):
        """Plot the the nodes and edges of the graph."""
        self.env.plot()
        plt.scatter(self.x_init[0], self.x_init[1], color="r", label="x_init")
        plt.scatter(self.samples[1:, 0], self.samples[1:, 1])
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
