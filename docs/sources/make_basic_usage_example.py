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

example.render("basic_usage_example.png", format="png", cleanup=True)
