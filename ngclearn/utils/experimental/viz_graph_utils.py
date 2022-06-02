"""
NGCGraph graph visualization functions.
"""
import matplotlib.pyplot as plt
import networkx as nx
from pyvis import network as pvnet

def visualize_graph(model, output_dir=None, height='500px', width='500px'):
    """
    Generates a graphical plot of the argument NGCGraph system.
    Note that a dynamic HTML object will be generated where the user can
    manipulate the graph to adhere to their own aesthetic constraints. Note
    that this will require opening up the generated .html object and
    then saving the final created graph by right-clicking and saving the
    final PNG.

    Args:
        model: the NGCGraph object to generate a network graph visualization of

        output_dir: the path/directory to save the GraphML output

        height: height of pyviz HTML graph (in px) to directly alter

        width: width of pyviz HTML graph (in px) to directly alter
    """
    output_dir_ = output_dir
    if output_dir_ is None:
        output_dir_ = ""
    Gx = nx.DiGraph() #MultiDiGraph
    G = pvnet.Network(directed=True, height=height, width=width)
    '''
    # Note: ngc-learn color/symbol coding scheme:
    blue (red if learnable) = dense cable, weight/width = 3
    blue (red if learnable) = simple cable, weight/width = 1.5, dashes = True
    antiquewhite = spiking node
    gainsboro = state node
    lavender = forward node
    mistyrose = error node
    '''
    node_id = 0
    name_id_table = {}
    for node_name in model.nodes: # extract the nodes
        node = model.nodes[node_name]
        name_id_table[node_name] = node_id
        node_type = node.node_type
        if node_type == "error":
            G.add_node(node_id, label=node_name, color="mistyrose", shape='ellipse', font='16px arial black')
            Gx.add_node(node_id, label=node_name, color="mistyrose", shape='ellipse', font='16px arial black')
        elif node_type == "state":
            G.add_node(node_id, label=node_name, color="gainsboro", shape='ellipse', font='14px arial')
            Gx.add_node(node_id, label=node_name, color="gainsboro", shape='ellipse', font='14px arial')
        elif "spike" in node_type:
            G.add_node(node_id, label=node_name, color="antiquewhite", shape='ellipse', font='14px arial')
            Gx.add_node(node_id, label=node_name, color="gainsboro", shape='ellipse', font='14px arial')
        elif "forward" in node_type:
            G.add_node(node_id, label=node_name, color="lavender", shape='ellipse', font='14px arial')
            Gx.add_node(node_id, label=node_name, color="gainsboro", shape='ellipse', font='14px arial')
        node_id += 1

    for cable_name in model.cables: # extract the edges (cables)
        cable = model.cables[cable_name]
        src_node = cable.src_node.name #cable.src_comp
        dest_node = cable.dest_node.name
        short_name = cable.short_name

        cable_type = cable.cable_type
        is_learnable = cable.is_learnable
        edge_color = "blue"
        src_node = name_id_table[src_node]
        dest_node = name_id_table[dest_node]
        if is_learnable == True:
            edge_color = "red"
        if cable_type == "simple":
            #simple_short_name = "{}".format(cable.coeff)
            G.add_edge(src_node, dest_node, label=short_name, weight=1.5, width=1.5,
                       color=edge_color, dashes=True)
            Gx.add_edge(src_node, dest_node, label=short_name, weight=1.5, width=1.5,
                       color=edge_color, dashes=True)
        elif cable_type == "dense":
            G.add_edge(src_node, dest_node, label=short_name, weight=3, width=3,
                       color=edge_color)
            Gx.add_edge(src_node, dest_node, label=short_name, weight=3, width=3,
                       color=edge_color)
    #g = G.copy() # some attributes added to nodes
    #net = pvnet.Network(directed=True, height=height, width=width)
    #net.from_nx(g)
    G.show_buttons(filter_=['physics'])
    G.show('{}{}.html'.format(output_dir_, model.name))
    # save networkx graph to GraphML format for external use
    nx.write_graphml_lxml(Gx, "{}{}.graphml".format(output_dir_, model.name))
    return Gx
