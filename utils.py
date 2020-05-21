# Standard libraries
import matplotlib.pyplot as plt
import numpy as np

# Custom libraries
import shapes


class GeoTools:
    """Utility class for geometry operations.
    """

    @staticmethod
    def onSegment(p, q, r):
        """Return true if Point q lies on line segment 'pr'."""

        if (
            (q.x <= max(p.x, r.x))
            and (q.x >= min(p.x, r.x))
            and (q.y <= max(p.y, r.y))
            and (q.y >= min(p.y, r.y))
        ):
            return True
        return False

    @staticmethod
    def orientation(p, q, r):
        """Return the orientation of an ordered triplet (p,q,r)."""

        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
        if val > 0:  # CW orientation
            return 1
        elif val < 0:  # CCW orientation
            return 2
        else:  # Colinear orientation
            return 0

    @staticmethod
    def doIntersect(p1, q1, p2, q2):
        """Return true if the line segments 'p1q1' and 'p2q2' intersect."""

        o1 = GeoTools.orientation(p1, q1, p2)
        o2 = GeoTools.orientation(p1, q1, q2)
        o3 = GeoTools.orientation(p2, q2, p1)
        o4 = GeoTools.orientation(p2, q2, q1)

        # General case
        if (o1 != o2) and (o3 != o4):
            return True

        # Special Cases
        # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
        if (o1 == 0) and GeoTools.onSegment(p1, p2, q1):
            return True
        # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
        if (o2 == 0) and GeoTools.onSegment(p1, q2, q1):
            return True
        # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
        if (o3 == 0) and GeoTools.onSegment(p2, p1, q2):
            return True
        # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
        if (o4 == 0) and GeoTools.onSegment(p2, q1, q2):
            return True
        return False

    @staticmethod
    def array2point(array):
        return shapes.Point(array[0], array[1])

    @staticmethod
    def sample_in_range(x_range):
        """Sample a node uniformly in the state space."""
        return (x_range[:, 1] - x_range[:, 0]) * np.random.uniform(
            size=x_range.shape[0]
        ) + x_range[:, 0]

    @staticmethod
    def get_mid_point(edge):
        return edge[0] + (edge[1] - edge[0]) / 2

    @staticmethod
    def get_normal_vect(edge):
        pA, pB = edge
        horizontal = abs(pA.y - pB.y) < 1e-05
        if horizontal:
            v = np.array([-(pB.y - pA.y) / (pB.x - pA.x), 1])
        else:
            v = np.array([1, -(pB.x - pA.x) / (pB.y - pA.y)])
        v /= np.linalg.norm(v)
        return v


class TestUtil:
    @staticmethod
    def check_square(mat):
        assert mat.shape[0] == mat.shape[1]

    @staticmethod
    def check_symmetric(mat, rtol=1e-05, atol=1e-08):
        TestUtil.check_square(mat)
        assert np.allclose(mat, mat.T, rtol=rtol, atol=atol)

    @staticmethod
    def check_same_size(vect_1, vect_2):
        assert vect_1.shape[0] == vect_2.shape[0]


class PlotTools:
    """Utility class for plotting operations.
    """

    @staticmethod
    def plot_min_proj(polygons, p):
        plt.figure(figsize=(10, 10))
        for polygon in polygons:
            polygon.plot_min_proj(p)
