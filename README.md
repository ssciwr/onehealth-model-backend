# oneHealth-Model backend 
## Description
This repository consists of a collection of models that share the same infrastructure code they are built upon. 


## Installation 
Download the repository 

```bash
git clone https://github.com/ssciwr/onehealth-model-backend.git
```

go to the base directory of the repository, and run 

```bash
pip install .
```
it is strongly recommended to use a virtual environment to install packages into. This will keep the package self-contained without its dependencies polluting the system.

In order to be able to visualize the computational graphs of the models you use or build, you need to install the package with the `viz` option. This will install the graphviz package which will take care of the visualization. 

```bash
pip install .[viz]
```
or, on macos: 

```bash
pip install ".[viz]"
```
Graphviz itself has additional dependencies it needs to install. For more details, see [here](https://github.com/xflr6/graphviz?tab=readme-ov-file#installation). 


## Troubleshooting
to be done.
