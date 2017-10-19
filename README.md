CALOUR
======

[![Build Status](https://travis-ci.org/biocore/calour.png?branch=master)](https://travis-ci.org/biocore/calour)
[![Coverage Status](https://coveralls.io/repos/github/biocore/calour/badge.svg?branch=master)](https://coveralls.io/github/biocore/calour?branch=master)

exploratory and interactive microbiome analysis based on heatmaps

Install
=======
Create a [conda](http://conda.pydata.org/docs/install/quick.html) environment for calour:
```
conda create -n calour python=3.5 matplotlib numpy scipy pandas qt jupyter scikit-learn statsmodels
```

and activate it using:
```
source activate calour
```

Install dependecies of biom-format and scikit-bio:
```
conda install -c biocore biom-format scikit-bio
```

Install calour:
```
pip install git+git://github.com/biocore/calour.git
```

Install database interfaces (optional)
--------------------------------------
* Install the [dbBact](http://www.dbbact.org) calour interface:
```
pip install git+git://github.com/amnona/dbbact-calour
```


* Install the [phenotype-database](https://doi.org/10.6084/m9.figshare.4272392) calour interface:

(based on : [Hiding in Plain Sight: Mining Bacterial Species Records for Phenotypic Trait Information](http://msphere.asm.org/content/2/4/e00237-17) - Barberán et al. 2017)
```
pip install git+git://github.com/amnona/pheno-calour
```

* For metabolomics, also install the [GNPS](http://gnps.ucsd.edu/) calour interface:
```
pip install git+git://github.com/amnona/gnps-calour
```

Install additional user interfaces
----------------------------------

If you use calour in Jupyter Notebook, it is highly recommended to
install [ipywidgets](https://github.com/jupyter-widgets/ipywidgets):
```
conda install -c conda-forge ipywidgets
```
or
```
pip install ipywidgets
```

If you would like to use the graphical user interface, you will need to install
the GUI interface [EZCalour](https://github.com/amnona/EZCalour):
```
pip install git+git://github.com/amnona/EZCalour
```


Use Calour
==========

Full documentation is located
[here](http://biocore.github.io/calour/). One strength of Calour is
that users can interactivelly explore the data patterns on the
heatmap. The key and mouse instructions to explore and manuvor the heatmap is explained
[here](http://biocore.github.io/calour/generated/calour.heatmap.plot.html#calour.heatmap.plot)

You can also check out a very simple demo of Calour usage in [this
Jupyter
Notebook](https://github.com/biocore/calour/blob/master/notebooks/demo.ipynb).
A real use case of Calour on a real microbiome data set is also shown
in [this Jupyter
Notebook](https://github.com/biocore/calour/blob/master/notebooks/.ipynb).


