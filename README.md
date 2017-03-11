CALOUR
======

[![Build Status](https://travis-ci.org/amnona/calour.png?branch=master)](https://travis-ci.org/amnona/calour)
[![Coverage Status](https://coveralls.io/repos/github/amnona/calour/badge.svg?branch=master)](https://coveralls.io/github/amnona/calour?branch=master)

exploratory and interactive microbiome analysis based on heatmaps

Install
=======
Create a [conda](http://conda.pydata.org/docs/install/quick.html) environment for calour:
```
conda create -n calour python=3.5 matplotlib numpy scipy pandas qt jupyter scikit-learn
```

and activate it using:
```
source activate calour
```

Install biom-format and scikit-bio:
```
conda install -c biocore biom-format scikit-bio
```

Install calour:
```
conda install -c zechxu calour
```

or (for the latest develop version):
```
pip install git+git://github.com/amnona/calour.git
```

Install the [dbBact](http://www.dbbact.org) calour interface:
```
pip install git+git://github.com/amnona/dbbact-calour
```

Optionally, can also install the calour GUI interface [EZCalour](https://github.com/amnona/EZCalour):

Using calour
============
Calour can be used from within an ipython session / jupyter notebook or as a standalone GUI (EZCalour).

A sample jupyter notebook is located in:
[https://github.com/amnona/calour/blob/master/notebooks/demo.ipynb](https://github.com/amnona/calour/blob/master/notebooks/demo.ipynb)

Full Documentation is located in:
[http://biocore.github.io/calour/](http://biocore.github.io/calour/)


Keys and mouses instructions is [here](http://biocore.github.io/calour/generated/calour.heatmap.plot.html#calour.heatmap.plot)

