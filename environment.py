# Standard libraries
import matplotlib.pyplot as plt


class Environment2D:
    """Environment2D class.
    Instantiated with the range in x and in y.
    Its building blocks are Rectangle objects.
    """

    def __init__(self, x_lims, y_lims):
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.initialize_rectangles()

    def initialize_rectangles(self):
        self.rectangles = {}

    def clear(self):
        self.initialize_rectangles()

    def add_rectangles(self, list_rectangles):
        for rectangle in list_rectangles:
            self.add_rectangle(rectangle)

    def add_rectangle(self, rectangle):
        try:
            assert rectangle.id not in self.rectangles
            self.rectangles[rectangle.id] = rectangle
        except AssertionError:
            print(
                "ID {} already exists inside environment! Please choose another ID.".format(
                    rectangle.id
                )
            )
            raise

    def contains(self, p):
        for rectangle_id in self.rectangles:
            if self.rectangles[rectangle_id].contains(p):
                return True
        return False

    def is_intersected(self, line, exclude=-1):
        for rectangle_id in self.rectangles:
            if rectangle_id == exclude:
                continue
            elif self.rectangles[rectangle_id].is_intersected(line):
                return True
        return False

    def get_lines_seen(self, p):
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
            lines_possibly_seen = rectangle.get_lines_possibly_seen(p)
            # Get rid of false positives
            lines_seen = self.curate_lines(lines_possibly_seen, rectangle_id, p)
            if lines_seen:
                lines_seen_dict[rectangle_id] = lines_seen
        return lines_seen_dict

    def curate_lines(self, lines_possibly_seen, rectangle_id, p):
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
            edge = self.rectangles[rectangle_id].edges[edge_idx]
            p1, p2 = edge[0], edge[1]
            if not self.intersects_triangle(p1, p2, p, exclude=rectangle_id):
                lines_seen.append(edge_idx)
        return lines_seen

    def intersects_triangle(self, p1, p2, p3, exclude):
        """Return True if the triangle (p1, p2, p3) collides
        with any of the obstacles contained in self, with the exception
        of one which we choose to exclude.
        """
        for rectangle_id in self.rectangles:
            if rectangle_id == exclude:
                continue
            elif self.rectangles[rectangle_id].intersects_triangle(p1, p2, p3):
                return True
        return False

    def plot(self, figsize=(10, 10)):
        plt.xlim(self.x_lims)
        plt.ylim(self.y_lims)
        for rectangle_id, rectangle in self.rectangles.items():
            rectangle.plot()
        plt.gca().set_aspect("equal", adjustable="box")

    def plot_min_proj(self, p):
        self.plot()
        for rectangle_id, rectangle in self.rectangles.items():
            p.plot()
            min_dist, min_proj = rectangle.get_min_dist(p)
            min_proj.plot("mo", size=4)
            p.plot_edge(min_proj)
