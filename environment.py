# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Custom libraries
from shapes import Rectangle
from utils import GeoTools


class Environment2D:
    """Environment2D class.
    Instantiated with the range in x and in y.
    Its building blocks are Rectangle objects.
    """

    def __init__(self, x_lims, y_lims):
        self.x_range = np.array([x_lims, y_lims])
        self.initialize()

    def initialize(self):
        self.rectangles = {}
        self.vol_tot = self.as_rectangle().as_poly["original"].volume
        self.vol_obs = 0
        self.vol_free = self.vol_tot
        self.nlines_tot = 0

    def clear(self):
        """Meant to be called outside the class.
        """
        self.initialize()

    def as_rectangle(self):
        """Return the environment as a Rectangle object."""
        return Rectangle(
            -1,
            self.x_range[0][0],
            self.x_range[1][0],
            self.x_range[0][1] - self.x_range[0][0],
            self.x_range[1][1] - self.x_range[1][0],
            0,
        )

    def add_rectangles(self, list_rectangles):
        """Add a list of rectangles to the rectangles attribute.
        Args:
            list_rectangles (list of Rectangle): list of rectangles to add.
        """
        for rectangle in list_rectangles:
            self.add_rectangle(rectangle)

    def add_rectangle(self, rectangle, verbose=True):
        """Add a rectangle to the rectangles attribute (list) if valid
        and update the range of the environment if said object goes beyond the
        current limits.
        Args:
            rectangle (Rectangle): rectangle to add.
        """
        try:
            assert self.is_valid_id(rectangle)
        except AssertionError:
            print(
                "ID {} already exists inside environment! Please choose another ID.".format(
                    rectangle.id
                )
            )
            raise
        try:
            assert self.is_valid_rectangle(rectangle, verbose)
            self.rectangles[rectangle.id] = rectangle
            vertices = rectangle.get_vertices("worst", as_array=True)
            min_values = np.min(vertices, axis=0)
            max_values = np.max(vertices, axis=0)
            self.x_range[0][0], self.x_range[1][0] = (
                min(self.x_range[0][0], min_values[0] - 1),
                min(self.x_range[1][0], min_values[1] - 1),
            )
            self.x_range[0][1], self.x_range[1][1] = (
                max(self.x_range[0][1], max_values[0] + 1),
                max(self.x_range[1][1], max_values[1] + 1),
            )
            self.vol_tot = self.as_rectangle().as_poly["original"].volume
            self.vol_obs += rectangle.as_poly["full"].volume
            self.vol_free = self.vol_tot - self.vol_obs
            self.nlines_tot += 4
        except AssertionError:
            if verbose:
                print("Please choose another rectangle.")
            raise

    def is_valid_id(self, rectangle):
        """Return True if rectangle has an ID that is not taken already.
        Args:
            rectangle (Rectangle)
        """
        return rectangle.id not in self.rectangles

    def is_valid_rectangle(self, rectangle, verbose=True):
        """Return True if rectangle is a valid region in the environment, assuming
        by the "worst" (i.e. most inflated) line configurations.
        Args:
            rectangle (Rectangle)
        """
        return GeoTools.is_valid_region(rectangle.as_poly["worst"], self, verbose)

    def add_obstacles(self, num_obs, lgth, wdth, ang=None, max_iter=1e3):
        """Add num_obs obstacles of certain length and width, within
        the limits of the current environment.
        If ang is not specified, will pick a random rectangle angle.
        """
        max_id = max(self.rectangles)
        added = 0
        it = 0
        while added < num_obs:
            it += 1
            if it > max_iter:
                print("Could fit {} such rectangles in the environment.".format(added))
                break
            x, y = GeoTools.sample_in_range(self.x_range)
            if ang is None:
                rectangle = Rectangle(
                    max_id + added + 1, x, y, lgth, wdth, np.random.uniform(0, 360)
                )
            else:
                rectangle = Rectangle(max_id + added + 1, x, y, lgth, wdth, ang)
            try:
                assert rectangle.as_poly <= self.as_rectangle().as_poly
                self.add_rectangle(rectangle, verbose=False)
            except AssertionError:
                continue
            added += 1

    def contains(self, p, config="worst"):
        """Return True if self contains a point, assuming a certain configuration.
        Args:
            p (Point): point to check for.
            config (str): map configuration to check point containment for, defaults to
                "worst", but other options are "best" or "full".
        """
        for rectangle_id in self.rectangles:
            if self.rectangles[rectangle_id].contains(p, config):
                return True
        return False

    def is_intersected(self, line, exclude=-1, config="worst"):
        """Return True if self is intersected by a line, assuming a certain
        configuration.
        Args:
            line (list of Point): line to check for.
            exclude (int): rectangle ID to exclude for this intersection check.
            config (str): map configuration to check point containment for, defaults to
                "worst", but other options are "best" or "full".
        """
        for rectangle_id in self.rectangles:
            if rectangle_id == exclude:
                continue
            elif self.rectangles[rectangle_id].is_intersected(line, config):
                return True
        return False

    def get_lines_seen(self, p, config):
        """Return the lines of self.rectangles that are
        entirely seen by a point p, in the format:
        {
            rectangle_id_1: [line_idx_1, line idx_2, ...],
            rectangle_id_2: [line_idx_3, ...]
        }
        """
        lines_seen_dict = {}
        for rectangle_id in self.rectangles:
            rectangle = self.rectangles[rectangle_id]
            # All the potentially seen lines
            lines_possibly_seen = rectangle.get_lines_possibly_seen(p, config)
            # Get rid of false positives
            lines_seen = self.curate_lines(lines_possibly_seen, rectangle_id, p, config)
            if lines_seen:
                lines_seen_dict[rectangle_id] = lines_seen
        return lines_seen_dict

    def curate_lines(self, lines_possibly_seen, rectangle_id, p, config):
        """Helper function for self.get_lines_seen.
        Return the lines actually seen by a point p by checking if:
            for each line in lines_possibly_seen:
                the triangle formed by (p1, p2, p), where p1 and p2 are the
                endpoints of line, collides with any of the obstacles
                defined in self, with the exception of the one with the
                id rectangle_id.
        """
        lines_seen = []
        for edge_idx in lines_possibly_seen:
            edge = self.rectangles[rectangle_id].edges[config][edge_idx]
            p1, p2 = edge[0], edge[1]
            if not self.intersects_triangle(
                p1, p2, p, exclude=rectangle_id, config=config
            ):
                lines_seen.append(edge_idx)
        return lines_seen

    def intersects_triangle(self, p1, p2, p3, exclude, config):
        """Return True if the triangle (p1, p2, p3) collides
        with any of the obstacles contained in self, with the exception
        of one which we choose to exclude.
        """
        for rectangle_id in self.rectangles:
            if rectangle_id == exclude:
                continue
            elif self.rectangles[rectangle_id].intersects_triangle(p1, p2, p3, config):
                return True
        return False

    def plot(self, show_error_bounds=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        plt.xlim(self.x_range[0])
        plt.ylim(self.x_range[1])
        for rectangle_id, rectangle in self.rectangles.items():
            rectangle.plot(ax=ax, show_error_bounds=show_error_bounds)
        plt.gca().set_aspect("equal", adjustable="box")

    def plot_min_proj(self, p):
        self.plot()
        for rectangle_id, rectangle in self.rectangles.items():
            p.plot()
            min_dist, min_proj = rectangle.get_min_dist(p)
            min_proj.plot("mo", size=4)
            p.plot_edge(min_proj)
