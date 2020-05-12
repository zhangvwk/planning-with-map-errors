# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.constants import degree

# Custom libraries
from utils import GeoTools


class Point:
    """2D Point class.
    Supports basic arithmetic and euclidean operations.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, p2):
        return Point(self.x + p2.x, self.y + p2.y)

    def __mul__(self, c):
        return Point(self.x * c, self.y * c)

    def __rmul__(self, c):
        return self * c

    def __truediv__(self, c):
        return self * (1.0 / c)

    def __neg__(self):
        return -1 * self

    def __sub__(self, p2):
        return self + (-p2)

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def norm(self):
        return np.linalg.norm((self.x, self.y))

    def dot(self, p2):
        return np.array((self.x, self.y)).dot(np.array((p2.x, p2.y)))

    def is_in_triangle(self, p1, p2, p3):
        o1 = GeoTools.orientation(self, p2, p1)
        o2 = GeoTools.orientation(self, p3, p2)
        o3 = GeoTools.orientation(self, p1, p3)
        all_CCW = o1 == 2 and o2 == 2 and o3 == 2
        all_CW = o1 == 1 and o2 == 1 and o3 == 1
        return all_CCW or all_CW

    def plot(self, c="ro", size=8):
        plt.plot(self.x, self.y, c, markersize=size)

    def plot_edge(self, p2):
        plt.plot([self.x, p2.x], [self.y, p2.y], linestyle="--", color="k", alpha=0.75)


class Polygon:
    """Polygon class.
    Instantiated with an id and a list of Point objects.
    """

    def __init__(self, id_, vertices, plot_sty="ggplot"):
        try:
            self.id = id_
            self.vertices = vertices
            self.store_edges()
            assert self.is_valid()
            self.x_list, self.y_list = map(
                list, zip(*[(v.x, v.y) for v in self.vertices])
            )
            mpl.style.use(plot_sty)
        except AssertionError:
            print("Polygon is not valid.")
            raise

    def get_vertices(self, as_array=True):
        if not as_array:
            return self.vertices
        else:
            return np.array([np.array([v.x, v.y]) for v in self.vertices])

    def store_edges(self):
        self.edges = []
        V = len(self.vertices)
        for i in range(V):
            self.edges.append((self.vertices[i], self.vertices[(i + 1) % V]))

    def is_valid(self):
        for i in range(len(self.edges) - 2):
            for j in range(i + 2, len(self.edges) - (i == 0)):
                edge_i = self.edges[i]
                edge_j = self.edges[j]
                if GeoTools.doIntersect(edge_i[0], edge_i[1], edge_j[0], edge_j[1]):
                    return False
        return True

    def is_intersected(self, line):
        for edge in self.edges:
            if GeoTools.doIntersect(edge[0], edge[1], line[0], line[1]):
                return True
        return False

    def intersects_triangle(self, p1, p2, p3):
        """Return true if self intersects with triangle formed by (p1, p2, p3).
        First check if a vertex of self is within the triangle.
        Second check if an edge of self intersects with the triangle.
        """
        for p in self.vertices:
            if p.is_in_triangle(p1, p2, p3):
                return True
        for triangle_edge in [[p1, p2], [p1, p3], [p2, p3]]:
            if self.is_intersected(triangle_edge):
                return True
        return False

    def plot(self, show_edges_id=True):
        plt.fill(self.x_list, self.y_list, alpha=0.75, label=self.id)
        for edge_idx in range(len(self.edges)):
            edge = self.edges[edge_idx]
            mid_point = edge[0] + (edge[1] - edge[0]) / 2
            plt.annotate(
                str(edge_idx), [mid_point.x, mid_point.y], fontsize=10, alpha=0.75
            )
        plt.legend(loc="best", fontsize=20)

    def get_min_dist(self, p):
        min_dist = float("inf")
        min_proj = None
        for edge in self.edges:
            v = edge[0]
            w = edge[1]
            l2 = (v - w).norm() ** 2
            if l2 == 0.0:
                min_dist = min(min_dist, (p - v).norm())
            else:
                t = max(0, min(1, (p - v).dot(w - v) / l2))
                proj = v + t * (w - v)
                new_dist = (p - proj).norm()
                if new_dist < min_dist:
                    min_dist = new_dist
                    min_proj = proj
        return min_dist, min_proj

    def plot_min_proj(self, p, sty="ggplot"):
        self.plot()
        p.plot()
        min_dist, min_proj = self.get_min_dist(p)
        min_proj.plot("mo", size=4)
        p.plot_edge(min_proj)


class Rectangle(Polygon):
    """Rectangle class (subclass of Polygon).
    Instantiated with a the 2D coordinates of a point, an angle
    about which to rotate the rectangle, a length, and a width.
    The angle is the direction along the length, a 0 angle means
    its direction is positive in the x axis. [deg]
    """

    def __init__(self, id_, x, y, lgth, wdth, angle):
        assert id_ >= 0
        self.id = id_
        self.x = x
        self.y = y
        self.lgth = lgth
        self.wdth = wdth
        self.yaw = angle * degree
        self.vertices = self.compute_vertices()
        super().__init__(self.id, self.vertices)
        self.errors = [[0, 0] for _ in range(4)]

    def compute_vertices(self):
        """Return the vertices in CCW order."""
        return [
            Point(self.x, self.y),
            Point(
                self.x + self.lgth * np.cos(self.yaw),
                self.y + self.lgth * np.sin(self.yaw),
            ),
            Point(
                self.x + self.lgth * np.cos(self.yaw) - self.wdth * np.sin(self.yaw),
                self.y + self.lgth * np.sin(self.yaw) + self.wdth * np.cos(self.yaw),
            ),
            Point(
                self.x - self.wdth * np.sin(self.yaw),
                self.y + self.wdth * np.cos(self.yaw),
            ),
        ]

    def set_errors(self, line_i, bound_l, bound_r):
        assert line_i < 4
        self.errors[line_i] = [bound_l, bound_r]

    def contains(self, p):
        A, B, C = self.vertices[:3]
        return (0 <= (B - A).dot(p - A) <= (B - A).norm() ** 2) and (
            0 <= (C - B).dot(p - B) <= (C - B).norm() ** 2
        )

    def get_lines_possibly_seen(self, p):
        num_edges = len(self.edges)
        for edge_idx in range(num_edges):
            edge_idx_prev = (edge_idx - 1) % num_edges
            edge_idx_next = (edge_idx + 1) % num_edges
            edge_curr = self.edges[edge_idx]
            edge_next = self.edges[edge_idx_next]
            dot_prod = (edge_curr[1] - edge_curr[0]).dot(p - edge_curr[0])
            norm_sq = (edge_curr[1] - edge_curr[0]).norm() ** 2
            facing_edge = 0 <= dot_prod <= norm_sq
            on_right = dot_prod > norm_sq
            on_correct_side = (edge_next[1] - edge_next[0]).dot(p - edge_next[0]) <= 0
            if facing_edge:
                if on_correct_side:
                    return [edge_idx]
                else:
                    return [(edge_idx + 2) % num_edges]
            if on_correct_side:
                if on_right:
                    return [edge_idx, edge_idx_next]
                else:
                    return [edge_idx_prev, edge_idx]
