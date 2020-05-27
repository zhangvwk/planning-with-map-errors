# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.constants import degree
import polytope as pc
from bitarray import frozenbitarray

# Custom libraries
from utils import GeoTools
from zonotope import Zonotope


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

    def as_array(self):
        return np.array([self.x, self.y])

    def norm(self):
        return np.linalg.norm(self.as_array())

    def dot(self, p2):
        return self.as_array().dot(p2.as_array())

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
            self.edges = GeoTools.vertices2edges(vertices)
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
            return GeoTools.vertices2array(self.vertices)

    def is_valid(self):
        for i in range(len(self.edges) - 2):
            for j in range(i + 2, len(self.edges) - (i == 0)):
                edge_i = self.edges[i]
                edge_j = self.edges[j]
                if GeoTools.doIntersect(edge_i[0], edge_i[1], edge_j[0], edge_j[1]):
                    return False
        return True

    def is_intersected(self, line, config):
        if config is not None:
            edges = self.edges[config]
        else:
            edges = self.edges
        for edge in edges:
            if GeoTools.doIntersect(edge[0], edge[1], line[0], line[1]):
                return True
        return False

    def intersects_triangle(self, p1, p2, p3, config=None):
        """Return true if self intersects with triangle formed by (p1, p2, p3).
        First check if a vertex of self is within the triangle.
        Second check if an edge of self intersects with the triangle.
        """
        if config is not None:
            vertices = self.vertices[config]
        else:
            vertices = self.vertices
        for p in vertices:
            if p.is_in_triangle(p1, p2, p3):
                return True
        for triangle_edge in [[p1, p2], [p1, p3], [p2, p3]]:
            if self.is_intersected(triangle_edge, config):
                return True
        return False

    def plot(self, show_id=True, show_edges_id=True, as_goal=False, mask=False):
        if not as_goal:
            if mask:
                plt.plot(
                    self.x_list + [self.x_list[0]],
                    self.y_list + [self.y_list[0]],
                    color="r",
                    alpha=0.75,
                )
            else:
                plt.fill(
                    self.x_list,
                    self.y_list,
                    alpha=0.5,
                    edgecolor="r",
                    linestyle="-",
                    label=self.id,
                )
            if show_edges_id:
                if isinstance(self.edges, dict):
                    edges = self.edges["original"]
                else:
                    edges = self.edges
                for edge_idx in range(len(edges)):
                    edge = edges[edge_idx]
                    mid_point = GeoTools.get_mid_point(edge)
                    plt.annotate(
                        str(edge_idx),
                        [mid_point.x, mid_point.y],
                        fontsize=10,
                        alpha=0.75,
                    )
            annot = str(self.id)
            color = "k"
        else:
            plt.fill(
                self.x_list,
                self.y_list,
                alpha=0.5,
                facecolor="lightsalmon",
                linestyle="-",
                edgecolor="red",
            )
            annot = "G"
            color = "r"
        plt.annotate(
            annot, self.get_center_gen()[0], fontsize=12, weight="bold", color=color
        )

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


CONFIG2BIT = {"worst": frozenbitarray("1111"), "best": frozenbitarray("0000")}
BIT2CONFIG = {frozenbitarray("1111"): "worst", frozenbitarray("0000"): "best"}


class Rectangle(Polygon):
    """Rectangle class (subclass of Polygon).
    Instantiated with a the 2D coordinates of a point, an angle
    about which to rotate the rectangle, a length, and a width.
    The angle is the direction along the length, a 0 angle means
    its direction is positive in the x axis. [deg]
    """

    def __init__(self, id_, x, y, lgth, wdth, angle):
        self.id = id_
        self.x = x
        self.y = y
        self.lgth = lgth
        self.wdth = wdth
        self.yaw = angle * degree
        self.vertices_original = self.compute_vertices()
        super().__init__(self.id, self.vertices_original)
        self.edges_original = self.edges
        self.error_bounds = np.zeros((4, 2))
        self.half_widths = np.zeros(4)
        self.center_offsets = np.zeros(4)
        self.actual_errors = np.zeros(4)
        self.update_stored_configs()

    def update_stored_configs(self):
        self.vertices = {
            "original": self.vertices_original,
            "full": self.get_vertices("full"),
            "worst": self.get_vertices("worst"),
            "best": self.get_vertices("best"),
            "actual": self.get_vertices("actual"),
        }
        self.edges = {
            "original": self.edges_original,
            "full": GeoTools.vertices2edges(self.vertices["full"]),
            "worst": GeoTools.vertices2edges(self.vertices["worst"]),
            "best": GeoTools.vertices2edges(self.vertices["best"]),
            "actual": GeoTools.vertices2edges(self.vertices["actual"]),
        }
        self.as_zono = {}
        self.as_zono = {
            "original": self.to_zonotope(),
            "full": self.to_zonotope("full"),
            "worst": self.to_zonotope("worst"),
            "best": self.to_zonotope("best"),
            "actual": self.to_zonotope("actual"),
        }
        self.as_poly = {}
        self.as_poly = {
            "original": self.to_poly(),
            "full": self.to_poly("full"),
            "worst": self.to_poly("worst"),
            "best": self.to_poly("best"),
            "actual": self.to_poly("actual"),
        }

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

    def get_vertices(self, config, as_array=False):
        """Return the four vertices given a config.
        config:
        None corresponds to the non-centered rectangle.
        "full" corresponds to the centered rectangle.
        "actual" corresponds to the actual rectangle.
        bitarray(b1,b2,b3,b4) where the b's are binary values corresponds to
        the rectangle with the left (0) or right (1) extremes for each line.
        "worst" is mapped to bitarray('1111').
        "best" is mapped to bitarray('0000').
        """
        vertices_config = self.vertices_original[:]
        if config is not None:
            for edge_idx in range(len(self.edges_original)):
                edge = self.edges_original[edge_idx]
                p1, p2 = edge
                v = GeoTools.get_normal_vect(edge)
                v = Point(v[0], v[1])
                p3, p4 = self.edges_original[(edge_idx + 1) % 4]
                # Make sure v is pointed outward wrt rectangle surface
                if (p2 - p4).dot(v) < 0:
                    v = -v
                if config == "full":
                    w = self.center_offsets[edge_idx]
                elif config == "actual":
                    w = self.actual_errors[edge_idx]
                else:
                    if not isinstance(config, frozenbitarray):
                        config = CONFIG2BIT[config]
                    w = self.error_bounds[edge_idx, int(config[edge_idx])]
                vertices_config[edge_idx] += w * v
                vertices_config[(edge_idx + 1) % 4] += w * v
        if as_array:
            return GeoTools.vertices2array(vertices_config)
        return vertices_config

    def set_line_error_bounds(self, line_i, bound_l, bound_r):
        assert line_i < 4
        assert bound_l <= bound_r
        self.error_bounds[line_i] = np.array([bound_l, bound_r])
        self.half_widths[line_i] = (bound_r - bound_l) / 2
        self.center_offsets[line_i] = -abs(bound_l) + self.half_widths[line_i]
        self.update_stored_configs()

    def set_error_bounds(self, bounds_l, bounds_r):
        assert np.all(bounds_l <= bounds_r)
        self.error_bounds[:, 0] = bounds_l
        self.error_bounds[:, 1] = bounds_r
        self.half_widths = (bounds_r - bounds_l) / 2
        self.center_offsets = -np.abs(bounds_l) + self.half_widths
        self.update_stored_configs()

    def set_line_actual_error(self, line_i, err):
        assert line_i < 4
        assert self.error_bounds[line_i][0] <= err <= self.error_bounds[line_i][1]
        self.actual_errors[line_i] = err
        self.update_stored_configs()

    def set_actual_errors(self, err):
        assert np.all(self.error_bounds[:, 0] <= err) and np.all(
            self.error_bounds[:, 1] >= err
        )
        self.actual_errors = err
        self.update_stored_configs()

    def contains(self, p, config="worst"):
        """Return True if Point object p lies within self."""
        if config in self.as_poly:
            rec_poly = self.as_poly[config]
        else:
            rec_poly = self.to_poly(config)
        A = rec_poly.A
        b = rec_poly.b
        return np.all(A.dot(p.as_array()) <= b)

    def is_intersected(self, line, config="worst"):
        for edge in self.edges[config]:
            if GeoTools.doIntersect(edge[0], edge[1], line[0], line[1]):
                return True
        return False

    def get_lines_possibly_seen(self, p, config):
        """Return the list of the subset of self.edges that are
        possibly seen by point p.
        """
        edges = self.edges[config]
        num_edges = len(edges)
        for edge_idx in range(num_edges):
            edge_idx_prev = (edge_idx - 1) % num_edges
            edge_idx_next = (edge_idx + 1) % num_edges
            edge_curr = edges[edge_idx]
            edge_prev = edges[edge_idx_prev]
            edge_next = edges[edge_idx_next]
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
                    if abs((p - edge_curr[1]).dot(edge_next[1] - edge_next[0])) < 1e-5:
                        return [edge_idx_next]
                    return [edge_idx, edge_idx_next]
                else:
                    if abs((p - edge_curr[1]).dot(edge_prev[1] - edge_prev[0])) < 1e-5:
                        return [edge_idx_prev]
                    return [edge_idx_prev, edge_idx]

    def to_poly(self, config=None):
        """Convert to a Polytope object."""
        if isinstance(config, str):
            try:
                return self.as_poly[config]
            except KeyError:
                pass
        else:
            try:
                return self.as_poly[BIT2CONFIG[frozenbitarray(config)]]
            except KeyError:
                pass
        if config in self.as_zono:
            return self.as_zono[config].to_poly()
        return self.to_zonotope(config).to_poly()

    def to_zonotope(self, config=None):
        """Convert to a Zonotope object given a config.
        """
        if isinstance(config, str):
            try:
                return self.as_zono[config]
            except KeyError:
                pass
        else:
            try:
                return self.as_zono[BIT2CONFIG[frozenbitarray(config)]]
            except KeyError:
                pass
        if config in self.vertices:
            vertices_config = self.vertices[config]
        else:
            vertices_config = self.get_vertices(config)
        center, generators = self.get_center_gen(vertices_config)
        return Zonotope(center, generators, np.zeros((2, 2)))

    def get_center_gen(self, vertices_config=None):
        if vertices_config is None:
            pA, pB, pC, _ = self.vertices_original
        else:
            pA, pB, pC, _ = vertices_config
        gen_1 = (pB - pA) / 2
        gen_2 = (pC - pB) / 2
        center = (pA + gen_1 + gen_2).as_array()
        generators = np.concatenate(([gen_1.as_array()], [gen_2.as_array()])).T
        return center, generators

    def get_error_bounds(self):
        e_l = np.zeros(4)
        e_r = np.zeros(4)
        for i in range(4):
            e_l[i], e_r[i] = self.error_bounds[i]
        return e_l, e_r

    def plot(
        self,
        show_id=True,
        show_edges_id=True,
        show_error_bounds=False,
        ax=None,
        as_goal=False,
    ):
        if show_error_bounds:
            A, b = self.as_poly["original"].A, self.as_poly["original"].b
            e_l, e_r = self.get_error_bounds()
            e = np.array(self.actual_errors)
            pc.Polytope(A, b + e_l).plot(ax=ax, alpha=0.2, linewidth=2, color="dimgrey")
            pc.Polytope(A, b + e_r).plot(ax=ax, alpha=0.2, linewidth=2, color="dimgrey")
            pc.Polytope(A, b + e).plot(
                ax=ax, alpha=0.6, color="cornflowerblue", linewidth=2, linestyle="-"
            )
        super().plot(
            show_id=show_id, show_edges_id=show_edges_id, as_goal=as_goal, mask=True
        )
