# Getting started 


## What is `heiplanet-models` all about?
This package bundles the model implementation for the [heiplanet project](TODO) and provides a development platform for their maintenance and extension as well as the addition of new model code.
All models are implemented in Python using xarray, geopandas, and numpy, but additional dependencies can be added as needed. 


## Installation
If you want to use the models that exist in `heiplanet-models`, execute the following steps to install the package: 

- Make a new virtual environment, e.g., with python `venv`: 
```bash
python -m venv venv 
```
In this case, the new environment will be called `venv`. This can then be activated with: 

```bash 
source ./venv/bin/activate 
``` 
Virtual environment are a great way to bundle the dependencies of a project, e.g., `heiplanet-models` in one place without polluting your system's python distribution or intefering with dependencies of other projects. With virtual environments, you can have an arbitrary number of isolated projects running alongside each other without interference. 


- Install the package: 
To get the current release, after activating the virtual environment, type: 

```bash 
python -m pip install heiplanet-models
```
This will pull in the package and all its basic dependencies. 


## Installation as a developer
The steps for creating and activating a virtual environment stay the same. Execute those first. Then: 

- Download the repository 

```bash
git clone https://github.com/ssciwr/onehealth-model-backend.git
```

- After creating and activating a new virtual environment, go to the base directory of the repository, and run 

```bash
pip install -e .
```
This will install the version of the code on the current `main` branch in `editable` mode, such that changes you make are immediatelly reflected in the importable package.

In order to be able to visualize the computational graphs of the models you use or build, you need to install the package with the `viz` option. This will install the graphviz package which will take care of the visualization. 

```bash
pip install .[viz]
```
or, on macos: 

```bash
pip install ".[viz]"
```
Graphviz itself has additional dependencies it needs to install. For more details, see [here](https://github.com/xflr6/graphviz?tab=readme-ov-file#installation). 

## Using existing models
In order to use existing models, no coding is required, all you need to do is write a yaml file in which the model parameters are defined.
