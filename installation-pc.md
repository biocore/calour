CALOUR installation instructions for windows 10
===============================================

Install miniconda (if already installed - skip this step):
----------------------------------------------------------
Go to [https://conda.io/miniconda.html](https://conda.io/miniconda.html), download the Python 3.6 64 bit exe installer, and run it.

You can select all the default options in the installer.

Create the Calour conda environment
-----------------------------------
In the windows start menu, select "anaconda prompt". You will get a command prompt.

Create a [conda](http://conda.pydata.org/docs/install/quick.html) environment for calour:
```
conda create -n calour python=3.5 matplotlib numpy scipy pandas qt jupyter scikit-learn statsmodels
```
(answer 'y' to the installation question)

and activate it using:
```
activate calour
```

Install biom-format
-------------------
Try to run the command:
```
pip install biom-format
```
If it fails (Error and then red colored text), it means you need to install the Microsoft Build Tools 2015 as follows:
go to [http://landinghub.visualstudio.com/visual-cpp-build-tools](http://landinghub.visualstudio.com/visual-cpp-build-tools), download "Build tools 2015", and run the installer. You can select all default options.

Then retry:
```
pip install biom-format
```

Install scikit-bio dependency:
```
conda install -c biocore biom-format scikit-bio
```

Install calour
--------------
Try to run the command:
```
pip install git+git://github.com/biocore/calour.git
```

If if fails, it probably means you don't have git commands installed. Do the following:
Get git for windows from: [https://git-for-windows.github.io/](https://git-for-windows.github.io/)

(use windows default console)

then close the anaconda prompt window and re-open using the Start menu->"anaconda prompt"

type:
```
activate calour
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

Running Calour
==============
From the Start menu, select "anaconda prompt". In the command prompt, type:
```
activate calour
```
To use calour in a [jupyter notebook](http://jupyter.org/), type:
```
jupyter notebook
```

To use the calour GUI, type:
```
ezcalour.py
```

Using Calour
============
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
