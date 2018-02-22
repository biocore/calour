# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import abc
from collections import namedtuple
from skbio import TreeNode
import pandas as pd
import numpy


class Dendrogram(TreeNode):
    """ Stores data to be plotted as a dendrogram.

    A `Dendrogram` object is represents a tree in addition to the
    key information required to create a tree layout prior to
    visualization.  No layouts are specified within this class,
    since this serves as a super class for different tree layouts.

    Parameters
    ----------
    use_lengths: bool
        Specifies if the branch lengths should be included in the
        resulting visualization (default True).

    Attributes
    ----------
    length
    leafcount
    height
    depth

    Notes
    -----
    `length` refers to the branch length of a node to its parent.
    `leafcount` is the number of tips within a subtree. `height` refers
    to the longest path from root to the deepst leaf in that subtree.
    `depth` is the number of nodes found in the longest path.

    """
    def __init__(self, use_lengths=True, **kwargs):
        """ Constructs a Dendrogram object for visualization.
        """
        super().__init__(**kwargs)

    def _cache_ntips(self):
        """ Counts the number of leaves under each subtree."""
        for n in self.postorder():
            if n.is_tip():
                n.leafcount = 1
            else:
                n.leafcount = sum(c.leafcount for c in n.children)

    def update_geometry(self, use_lengths, depth=None):
        """Calculate tree node attributes such as height and depth.

        Parameters
        ----------
        use_lengths: bool
           Specify if the branch length should be incorporated into
           the geometry calculations for visualization.
        depth: int
           The number of nodes in the longest path from root to leaf.

        This is agnostic to scale and orientation.
        """
        if self.length is None or not use_lengths:
            if depth is None:
                self.length = 0
            else:
                self.length = 1

        self.depth = (depth or 0) + self.length

        children = self.children
        if children:
            for c in children:
                c.update_geometry(use_lengths, self.depth)
            self.height = max([c.height for c in children]) + self.length
            self.leafcount = sum([c.leafcount for c in children])

        else:
            self.height = self.length
            self.leafcount = self.edgecount = 1

    def coords(self, height, width):
        """ Returns coordinates of nodes to be rendered in plot.

        Parameters
        ----------
        height : int
            The height of the canvas.
        width : int
            The width of the canvas.
        Returns
        -------
        pd.DataFrame
            index : str
                Name of node.
            x : float
                x-coordinate of node.
            y : float
                y-coordinate of node.
            child(i) : str
                Name of ith child node in that specific node.
                in the tree.
            is_tip : str
                Specifies if the node is a tip in the treee.
        """
        self.rescale(width, height)
        result = {}
        for node in self.postorder():
            children = {'child%d' % i: n.name
                        for i, n in enumerate(node.children)}
            coords = {'x': node.x2, 'y': node.y2}
            is_tip = {'is_tip': node.is_tip()}
            result[node.name] = {**coords, **children, **is_tip}
        result = pd.DataFrame(result).T

        # reorder so that x and y are first
        cols = ['x', 'y'] + sorted(list(set(result.columns) - set(['x', 'y'])))
        return result.loc[:, cols]

    @abc.abstractmethod
    def rescale(self, width, height):
        pass


class UnrootedDendrogram(Dendrogram):
    """ Stores data to be plotted as an unrooted dendrogram.

    A `UnrootedDendrogram` object is represents a tree in addition to the
    key information required to create a radial tree layout prior to
    visualization.

    Parameters
    ----------
    use_lengths: bool
        Specifies if the branch lengths should be included in the
        resulting visualization (default True).

    Attributes
    ----------
    length
    leafcount
    height
    depth
    """
    def __init__(self, **kwargs):
        """ Constructs a UnrootedDendrogram object for visualization.

        Parameters
        ----------
        use_lengths: bool
            Specifies if the branch lengths should be included in the
            resulting visualization (default True).
        """
        super().__init__(**kwargs)

    @classmethod
    def from_tree(cls, tree, use_lengths=True):
        """ Creates an UnrootedDendrogram object from a skbio tree.

        Parameters
        ----------
        tree : skbio.TreeNode
            Input skbio tree

        Returns
        -------
        UnrootedDendrogram
        """
        for n in tree.postorder():
            n.__class__ = UnrootedDendrogram

        tree.update_geometry(use_lengths)
        return tree

    def rescale(self, width, height):
        """ Find best scaling factor for fitting the tree in the figure.

        This method will find the best orientation and scaling possible
        to fit the tree within the dimensions specified by width and height.

        Parameters
        ----------
        width : float
            width of the canvas
        height : float
            height of the canvas

        Returns
        -------
        best_scaling : float
            largest scaling factor in which the tree can fit in the canvas.

        Notes
        -----
        """
        angle = (2 * numpy.pi) / self.leafcount
        # this loop is a horrible brute force hack
        # there are better (but complex) ways to find
        # the best rotation of the tree to fit the display.
        best_scale = 0
        for i in range(60):
            direction = i / 60.0 * numpy.pi
            # TODO:
            # This function has a little bit of recursion.  This will
            # need to be refactored to remove the recursion.

            points = self.update_coordinates(1.0, 0, 0, direction, angle)
            xs, ys = zip(*points)
            # double check that the tree fits within the margins
            scale = min(float(width) / (max(xs) - min(xs)),
                        float(height) / (max(ys) - min(ys)))
            # TODO: This margin seems a bit arbituary.
            # will need to investigate.
            scale *= 0.95  # extra margin for labels
            if scale > best_scale:
                best_scale = scale
                mid_x = width / 2 - ((max(xs) + min(xs)) / 2) * scale
                mid_y = height / 2 - ((max(ys) + min(ys)) / 2) * scale
                best_args = (scale, mid_x, mid_y, direction, angle)

        self.update_coordinates(*best_args)
        return best_scale

    def update_coordinates(self, s, x1, y1, a, da):
        """ Update x, y coordinates of tree nodes in canvas.

        `update_coordinates` will recursively updating the
        plotting parameters for all of the nodes within the tree.
        This can be applied when the tree becomes modified (i.e. pruning
        or collapsing) and the resulting coordinates need to be modified
        to reflect the changes to the tree structure.

        Parameters
        ----------
        s : float
            scaling
        x1 : float
            x midpoint
        y1 : float
            y midpoint
        a : float
            angle (degrees)
        da : float
            angle resolution (degrees)

        Returns
        -------
        points : list of tuple
            2D coordinates of all of the nodes.

        Notes
        -----
        This function has a little bit of recursion.  This will
        need to be refactored to remove the recursion.
        """
        # Constant angle algorithm.  Should add maximum daylight step.
        x2 = x1 + self.length * s * numpy.sin(a)
        y2 = y1 + self.length * s * numpy.cos(a)
        (self.x1, self.y1, self.x2, self.y2, self.angle) = (x1, y1, x2, y2, a)
        # TODO: Add functionality that allows for collapsing of nodes
        a = a - self.leafcount * da / 2
        if self.is_tip():
            points = [(x2, y2)]
        else:
            points = []
            # TODO:
            # This function has a little bit of recursion.  This will
            # need to be refactored to remove the recursion.
            for child in self.children:
                # calculate the arc that covers the subtree.
                ca = child.leafcount * da
                points += child.update_coordinates(s, x2, y2, a + ca / 2, da)
                a += ca
        return points


Dimensions = namedtuple('Dimensions', ['x', 'y', 'height'])


class RootedDendrogram(Dendrogram):
    """ Stores data to be plotted as an rooted dendrogram.

    A `RootedDendrogram` object is represents a tree in addition to the
    key information required to create a radial tree layout prior to
    visualization.

    Parameters
    ----------
    use_lengths: bool
        Specifies if the branch lengths should be included in the
        resulting visualization (default True).

    Attributes
    ----------
    length
    leafcount
    height
    depth
    """

    def width_required(self):
        return self.leafcount

    @abc.abstractmethod
    def xcoords(self, scale, x1):
        pass

    @abc.abstractmethod
    def ycoords(self, scale, y1):
        pass

    def rescale(self, width, height):
        """ Update x, y coordinates of tree nodes in canvas.

        Parameters
        ----------
        scale : Dimensions
            Scaled dimensions of the tree
        x1 : int
            X-coordinate of parent
        """
        xscale = width / self.height
        yscale = height / self.width_required()
        scale = Dimensions(xscale, yscale, self.height)

        # y coords done postorder, x preorder, y first.
        # so it has to be done in 2 passes.
        self.update_y_coordinates(scale)
        self.update_x_coordinates(scale)
        return xscale

    def update_y_coordinates(self, scale, y1=None):
        """The second pass through the tree.  Y coordinates only
        depend on the shape of the tree and yscale.

        Parameters
        ----------
        scale : Dimensions
            Scaled dimensions of the tree
        x1 : int
            X-coordinate of parent
        """
        if y1 is None:
            y1 = self.width_required() * scale.y
        child_y = y1
        for child in self.children:
            child.update_y_coordinates(scale, child_y)
            child_y -= child.width_required() * scale.y
        (self.y1, self.y2) = self.ycoords(scale, y1)

    def update_x_coordinates(self, scale, x1=0):
        """For non 'square' styles the x coordinates will depend
        (a bit) on the y coodinates, so they should be done first.
        Parameters
        ----------
        scale : Dimensions
            Scaled dimensions of the tree
        x1 : int
            X-coordinate of parent
        """
        (self.x1, self.x2) = self.xcoords(scale, x1)
        for child in self.children:
            child.update_x_coordinates(scale, self.x2)


class SquareDendrogram(RootedDendrogram):

    def ycoords(self, scale, y1):
        cys = [c.y1 for c in self.children]
        if cys:
            y2 = (cys[0] + cys[-1]) / 2.0
        else:
            y2 = y1 - 0.5 * scale.y
        return (y2, y2)

    def xcoords(self, scale, x1):
        if self.is_tip():
            return (x1, (scale.height - (self.height - self.length)) * scale.x)
        else:
            # give some margins for internal nodes
            dx = scale.x * self.length * 0.95
            x2 = x1 + dx
            return (x1, x2)

    @classmethod
    def from_tree(cls, tree):
        """ Creates an SquareDendrogram object from a skbio tree.

        Parameters
        ----------
        tree : skbio.TreeNode
            Input skbio tree
        Returns
        -------
        SquareDendrogram
        """
        for n in tree.postorder(include_self=True):
            n.__class__ = SquareDendrogram
        tree.update_geometry(use_lengths=False)
        return tree


def _plot_dendrogram(ax_dendrogram, table, edges):
    """ Plots the actual dendrogram.
    Parameters
    ----------
    ax_dendrogram : matplotlib.axes.Axes
        Contains the matplotlib axes in which the dendrogram will be plotted.
    table : pandas.DataFrame
        Contain sample/feature labels along with table of values.
        Rows correspond to samples, and columns correspond to features.
    edges : pd.DataFrame
        (x,y) coordinates for edges in the heatmap.
    """
    offset = 0.5
    for i in range(len(edges.index)):
        row = edges.iloc[i]
        ax_dendrogram.plot([row.x0, row.x1],
                           [row.y0 - offset, row.y1 - offset], '-k')
    ax_dendrogram.set_ylim([- offset, table.shape[0] - offset])
    ax_dendrogram.set_yticks([])
    ax_dendrogram.set_xticks([])


def plot_tree(exp, tree, ax):
    '''Plot a tree into the given axes

    Parameters
    ----------
    exp: Experiment
    tree: skbio.TreeNode
        The tree to plot
    ax: matplotlib.axes.Axes
        The axes where to plot the tree to

    Returns
    -------
    Experiment`, :class:`skbio.TreeNode
        both filtered to have common nodes and Experiment reordered to match the tree
    '''
    # reorder the tree and exp to match (exp features are sorted according to tree)
    exp, tree = _match_tips(exp, tree)

    dendrogram_width = 20
    # get edges from tree
    t = SquareDendrogram.from_tree(tree)
    pts = t.coords(width=dendrogram_width, height=exp.shape[1])
    edges = pts[['child0', 'child1']]
    edges = edges.dropna(subset=['child0', 'child1'])
    edges = edges.unstack()
    edges = pd.DataFrame({'src_node': edges.index.get_level_values(1),
                          'dest_node': edges.values})
    edges['x0'] = [dendrogram_width - pts.loc[n].x for n in edges.src_node]
    edges['x1'] = [dendrogram_width - pts.loc[n].x for n in edges.dest_node]
    edges['y0'] = [pts.loc[n].y for n in edges.src_node]
    edges['y1'] = [pts.loc[n].y for n in edges.dest_node]

    _plot_dendrogram(ax, exp.get_data().transpose(), edges)

    return exp, tree


def _match_tips(exp, tree):
    """ Returns the contingency table and tree with matched tips.

    Sorts the columns of the contingency table to match the tips in
    the tree.  The ordering of the tips is in post-traversal order.
    If the tree is multi-furcating, then the tree is reduced to a
    bifurcating tree by randomly inserting internal nodes.
    The intersection of samples in the contingency table and the
    tree will returned.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leafs correspond to the features.

    Returns
    -------
    pd.DataFrame :
        Subset of the original contingency table with the common features.
    skbio.TreeNode :
        Sub-tree with the common features.

    Raises
    ------
    ValueError:
        Raised if `table` and `tree` have incompatible sizes.

    See Also
    --------
    skbio.TreeNode.bifurcate
    skbio.TreeNode.tips
    """
    tips = [x.name for x in tree.tips()]
    common_tips = list(set(tips) & set(exp.feature_metadata.index.values))
    _tree = tree.shear(names=common_tips)

    _tree.bifurcate()
    _tree.prune()

    sorted_features = [n.name for n in _tree.tips()]
    newexp = exp.filter_ids(sorted_features, axis=1)
    return newexp, _tree
