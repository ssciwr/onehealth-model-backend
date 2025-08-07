import graphviz

example = graphviz.Digraph(comment="basic_usage_example")


example.node("read_data")
example.node("initialize_parameters")
example.node("project_data_to_map")
example.node("compute_temperature_data")
example.node("compute_humidity_data")
example.node("compute_r0")
example.node("save_data")

example.edge("read_data", "project_data_to_map")
example.edge("initialize_parameters", "project_data_to_map")
example.edge("project_data_to_map", "compute_temperature_data")
example.edge("project_data_to_map", "compute_humidity_data")
example.edge("compute_temperature_data", "compute_r0")
example.edge("compute_humidity_data", "compute_r0")
example.edge("compute_r0", "save_data")

example.render("basic_usage_example", format="png", cleanup=True)


# example for creation of a new model
creation_example = graphviz.Digraph(comment="creation_example")

creation_example.node("load_data")
creation_example.node("add")
creation_example.node("multiply")
creation_example.node("subtract")
creation_example.node("affine")
creation_example.node("save")

creation_example.edge("load_data", "add")
creation_example.edge("load_data", "multiply")
creation_example.edge("add", "multiply")
creation_example.edge("multiply", "subtract")
creation_example.edge("add", "subtract")
creation_example.edge("subtract", "affine")
creation_example.edge("affine", "save")

creation_example.render("creation_example", format="png", cleanup=True)
