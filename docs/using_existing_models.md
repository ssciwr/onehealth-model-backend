# Using an exisiting model 
This section covers how to use an existing model, in `heiplanet-models`. 
If you haven't already, please first go back and read [the 'Basic design' section](./basic_design.md). 

To use an exiting model involves three steps: 

- Decide on the parameters and function arguments you want to modify in your run from the defaults. 
- Copy the default configuration file and replace the parameters in the copy with the ones you want to have. You can also give them as a dictionary in python directly and pass them to the constructor, but this is not recommended. Rather, follow the principle of separation of concerns and make use of separate configuration files.
- Load the configuration file in a python session, instantiate a `computation_graph` object and run it. 

This is best shown in an example, so please check out [Example of how to use an existing model in the notebooks section](./sources/model_jmodel_python.ipynb) which walks you through all the steps needed. 

## Best practices 
- Store the configuration next to the data you produce.
- Visualize the model when running it for the first time. This will help you understand what it does and helps verify that it is put together the right way. 
- Make a new directory for each experiment, and add date and time to its name or into the config to keep track of experiments