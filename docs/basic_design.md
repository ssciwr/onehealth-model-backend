# Basic design of model implementations

## Principles
There are two basic design principles on which all models in this package are built: 
- **functional programming**: You write and compose **functions**, not just statements. If you wish to know more about different programming paradigms, see [here](), and if you want an introduction to functional programming in itself, see [here](). Functional programming is a natural way of composing dynamical systems, and alleviates the model designer from having to deal with a lot of the intricacies of object-orientation in python. Most of all, however, it lends itself well to automated system composition, i.e., we can automatically build a directed acyclic graph of functions, in which any given function can depend on the result of a number of predecessors and itself be a dependency of a number of other functions.

- **Separation of parameterization and functionality**: This means that you write the model code in the form of a collection of functions, and parameterize it in accordance with th real world system you want to model. But instead of supplying the parameters and control flow in the code directly, you write a configuration file that defines and provides them. This helps us in multiple ways: 
    - Separation of concerns: Code is for model functionality, config files are for definition of a certain model instance. You can run a model without any changes to the code by just changing the parameters in the supplied config file. This also allows us to easily build different versions of the same model by providing multiple configuration files which define a different composition of the functions that make up the model. 
    - Documentation and reproducibility: The configuration file used to run a model is stored alongside the data it produces, so it's automatically documented which parameters produced which results, and what functional composition was utilized. 

## Directed acyclic graphs for model composition from functions 
Each model is implemented as a collection of functions, which within the python files are not connected into a running model. They can, however, be made up of calls to other functions. 
For example, a model might consist of functions:
    - `read_data`
    - `initialize_parameters`
    - `project_data_to_map`
    - `compute_temperature_data`
    - `compute_humidity_data`
    - `compute_r0`
    - `save_data`

When we think about how these functions interact to form the fully-fledged model, we arrive at a web of interdependencies with clear definition of which function depends on which others. This lets us reason about model composition by considering how data flows through the functions: 

For example, this could look like this: 
- `read_data` does not depend on any other function. It is a `source` of data.
- `initialize_parameters` does not depend on any other functions as well, so it, too, is a `source` of data. 
- `project_data_to_map` depends on `read_data` and `initialize_parameters` 
- `compute_temperature_data` depends on `project_data_to_map`
- `compute_humidity_data` depends on `project_data_to_map`
- `compute_r0` depends on humidity and temperature at any given grid point, so it depends on both `compute_humidity_data` and `compute_temperature_data`. 
- finally, `save_data` depends on `compute_r0`. No further function depends on `save_data`, so it is a `sink` for the produced data.

This set of interdependencies creates a graph in which each node is a function and each edge tells us on which other functions it depends. For clarity, we usually turn these edges around and interpret them as telling us to what other functions data flows from any given function. 