# Let's create a very simple data set:

from calour import Experiment
import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
exp = Experiment(np.array([[0,9], [7, 4]]), sparse=False,
                 sample_metadata=pd.DataFrame({'category': ['A', 'B'],
                                               'ph': [6.6, 7.7]},
                                              index=['s1', 's2']),
                 feature_metadata=pd.DataFrame({'motile': ['y', 'n']}, index=['otu1', 'otu2']))

# Let's then plot the heatmap:

fig, ax = plt.subplots()
exp.heatmap(sample_field='category', feature_field='motile', title='Fig 1 log scale', ax=ax)   # doctest: +SKIP

# By default, the color is plot in log scale. Let's say we would like to plot heatmap in normal scale instead of log scale:

fig, ax = plt.subplots()
norm = mpl.colors.Normalize()
exp.heatmap(sample_field='category', feature_field='motile', title='Fig 2 normal scale',
            norm=norm, ax=ax)             # doctest: +SKIP

# Let's say we would like to show the presence/absence of each
# OTUs across samples in heatmap. And we define presence as
# abundance larger than 4:

expbin = exp.binarize(4)
expbin.data
# array([[0, 1],
# [1, 0]])

# Now we have converted the abundance table to the binary
# table. Let's define a binary color map and use it to plot the
# heatmap:

# define the colors
cmap = mpl.colors.ListedColormap(['r', 'k'])
# create a normalize object the describes the limits of each color
norm = mpl.colors.BoundaryNorm([0., 0.5, 1.], cmap.N)
fig, ax = plt.subplots()
expbin.heatmap(sample_field='category', feature_field='motile', title='Fig 3 binary',
               cmap=cmap, norm=norm, ax=ax)         # doctest: +SKIP
