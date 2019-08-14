# Installation instructions for Calour

There are three ways to obtain Calour:

* Install locally on Max/Linux/Windows 10 computer

  This is the recommended method for long term Calour usage, as it enables full utilization of the computer resources, as well as easy upgrades. However, the installation process is more complicated \(full instructions are detailed below\).

* Run locally using a virtualbox image

  This enables easy installation, while still running locally on selected computer.

* Run remotely on the [mybinder server](https://mybinder.org/v2/gh/biocore/calour/master?filepath=doc%2Fsource%2Fnotebooks)

  Enables hands on trying of Calour without the need to install. However, the mybinder server is not intended for heavy duty work.

# Installation instructions for Mac/Linux/Windows 10

## EZCalour easy installer for Windows
If you have windows and would like to use calour via a full point and click GUI (EZCalour), you can download a windows installer from [sourceforge](https://sourceforge.net/projects/ezcalour/files/ezcalour_setup.exe/download).

Just download the ezcalour_installer.exe and run it :)

## Install Calour locally

You need to install miniconda (or anaconda) first. Go to [https://conda.io/miniconda.html](https://conda.io/miniconda.html), download the Python 3 64 bit installer and run it. You can select all the default options in the installer. And then you can follow the instruction to install from bioconda or from github repository:

### Install from bioconda channel

For Mac/Linux users, you can skip the following manual installation steps and directly install Calour and its required dependencies with conda:

```python
conda create -n calour python=3.5
source activate calour
conda install -c bioconda calour
```

### Install the latest manually from github repository

> _**Windows Note**_ In the windows start menu, select "anaconda prompt". You will get a command prompt.

Create a [conda](http://conda.pydata.org/docs/install/quick.html) environment for calour:

```
conda create -n calour python=3.5 matplotlib numpy scipy pandas qt jupyter scikit-learn statsmodels h5py
```

and activate it using:

```
activate calour           # on Windows
source activate calour    # on Mac/Linux
```

Install additional dependencies

Try to run the command:

```
pip install biom-format
```

> _**Windows Note**_ If it fails \(Error and then red colored text\) on Windows, it means you need to install the Microsoft Build Tools 2015 as follows:
>
> go to [http://landinghub.visualstudio.com/visual-cpp-build-tools](http://landinghub.visualstudio.com/visual-cpp-build-tools), download "Build tools 2015", and run the installer \(you can select all default options\).
>
> Then retry:
>
> ```
> pip install biom-format
> ```

Install scikit-bio and docrep dependency:

```
pip install scikit-bio docrep
```

Try to run the command to install the lastest Calour:

```
pip install git+git://github.com/biocore/calour.git
```

> _**Windows Note**_ If it fails, it probably means you don't have git commands installed. Do the following:
>
> Get git for Windows from: [https://git-for-windows.github.io/](https://git-for-windows.github.io/) and install it \(use windows default console\)
>
> then close the anaconda prompt window and re-open using the Start menu -&gt; "anaconda prompt"
>
> type:
>
> ```
> activate calour
> pip install git+git://github.com/biocore/calour.git
> ```

## Install database interfaces \(optional\)

* Install the [dbBact](http://www.dbbact.org) calour interface:

  ```
  pip install git+git://github.com/amnona/dbbact-calour
  ```

* Install the [phenotype-database](https://doi.org/10.6084/m9.figshare.4272392) calour interface:

  (based on : [Hiding in Plain Sight: Mining Bacterial Species Records for Phenotypic Trait Information](http://msphere.asm.org/content/2/4/e00237-17) - Barberán et al. 2017\)

  ```
  pip install git+git://github.com/amnona/pheno-calour
  ```

* For metabolomics, also install the [GNPS](http://gnps.ucsd.edu/) calour interface:
  ```
  pip install git+git://github.com/amnona/gnps-calour
  ```

## Install additional user interfaces \(optional\)

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

# Running Calour

> _**Windows Note**_ From the Start menu, select "anaconda prompt" to get command prompt.

To start, activate the virtual environment:

```
activate calour           # on Windows
source activate calour    # on Mac/Linux
```

To use calour in a [jupyter notebook](http://jupyter.org/), you can just open a notebook and start to import and use calour.

To use the calour GUI, type and run:

```
ezcalour.py
```

# Installing Calour using the virtualbox image

## Installing VirtualBox

If not installed, install VirtualBox on your computer. VirtualBox is availble for Windows/Mac and Linux, and can be obtainined from [https://www.virtualbox.org/wiki/Downloads](https://www.virtualbox.org/wiki/Downloads).

## Downloading the Calour virtualbox image

Download the Calour image from [http://www.mediafire.com/file/13f6vetjuquha6b/calour\_vm\_2018\_5\_2.ova](http://www.mediafire.com/file/13f6vetjuquha6b/calour_vm_2018_5_2.ova). Note this is a large file \(~3.5Gb\) so download can take some time.

## Importing the Calour image to virtualbox

Run VirtualBox, and select "open appliace" from the file menu. Choose the Calour image file downloaded in the previous step.

## Running Calour

After running the Calour virtual machine in VirtualBox, open a new terminal window \(using the desktop shortcut\). To run EZCalour, type:

```
ezcalour
```

To run a jupyter notebook, type:

```
jupyter notebook
```

Calour example notebooks and datasets are located at:

```
/home/calour/examples
```

**NOTE**: the root user and password for the virtual machine are both: calour

# Running Calour on a remote mybinder server

## Go to the Calour mybinder server

open [https://mybinder.org/v2/gh/biocore/calour/master?filepath=doc%2Fsource%2Fnotebooks](https://mybinder.org/v2/gh/biocore/calour/master?filepath=doc%2Fsource%2Fnotebooks)

## Viewing/running tutorial notebooks

Select the notebook \(\*.ipynb\) from the list

## Uploading new data

Select the "upload" button and upload the biom table/mapping files

## Creating a new analysis notebook

Select the "new" button. Alternatively, you can select the "Generic Analysis.ipynb" notebook as a convenient starting point for analyzing microbiome data using Calour

# Problems? Questions? Suggestions?
Please visit the [Calour forum](https://groups.google.com/forum/#!forum/calour-support) for support
