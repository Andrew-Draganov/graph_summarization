import numpy as np
from pyvis import network as net
import networkx as nx
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
import matplotlib.pyplot as plt
from itertools import count
import os


#####
# Below is for plotting matplotlib graphs
#####

def draw_full_graph(
        g,
        node_str,
        edge_str,
        queried_str,
        node_cmap=plt.cm.gist_stern,
        edge_cmap=plt.cm.gist_stern,
        file_name='network.png',
        directory='figures'
    ):
    pos = nx.spring_layout(g, seed=123)
    node_attributes = np.array(list(nx.get_node_attributes(g, node_str).values()))
    queries = np.array(list(nx.get_node_attributes(g, queried_str).values()))
    edge_attributes = np.array(list(nx.get_edge_attributes(g, edge_str).values()))

    min_ea = min(edge_attributes)
    max_ea = max(edge_attributes)
    fig = plt.figure(figsize=(12, 10))
    ec = nx.draw_networkx_edges(
        g,
        pos,
        alpha=0.5,
        width=[2 * ea / max_ea for ea in edge_attributes],
        edge_color=edge_attributes,
        edge_cmap=edge_cmap,
        edge_vmin=min_ea,
        edge_vmax=max_ea
    )
    ecb = plt.colorbar(ec)
    ecb.set_label(edge_str)

    max_na = max(node_attributes)
    queried_nodelist = np.array(g.nodes())[np.where(queries == 1)]
    queried_attributes = node_attributes[np.where(queries == 1)]
    nc = nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=queried_nodelist,
        node_color=queried_attributes, 
        node_size=100,
        edgecolors=[1, 0, 0],
        linewidths=2,
        cmap=node_cmap,
        vmin=0,
        vmax=max_na
    )
    other_nodelist = np.array(g.nodes())[np.where(queries == 0)]
    other_attributes = node_attributes[np.where(queries == 0)]
    nc = nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=other_nodelist,
        node_color=other_attributes, 
        node_size=100,
        cmap=node_cmap,
        vmin=0,
        vmax=max_na
    )

    ncb = plt.colorbar(nc)
    ncb.set_label(node_str)
    plt.axis('off')
    plt.savefig(os.path.join(directory, file_name))
    plt.clf()
    plt.close()


def draw_subgraph(
        g,
        node_str,
        edge_str,
        node_cmap=plt.cm.gist_stern,
        edge_cmap=plt.cm.gist_stern,
        file_name='network.png',
        directory='figures'
    ):
    pos = nx.spring_layout(g, seed=123)
    node_attributes = np.array(list(nx.get_node_attributes(g, node_str).values()))
    edge_attributes = np.array(list(nx.get_edge_attributes(g, edge_str).values()))

    min_ea = min(edge_attributes)
    max_ea = max(edge_attributes)
    plt.figure(figsize=(12, 10))
    ec = nx.draw_networkx_edges(
        g,
        pos,
        alpha=0.5,
        width=[2 * ea / max_ea for ea in edge_attributes],
        edge_color=edge_attributes,
        edge_cmap=edge_cmap,
        edge_vmin=min_ea,
        edge_vmax=max_ea
    )
    ecb = plt.colorbar(ec)
    ecb.set_label(edge_str)

    max_na = max(node_attributes)
    nc = nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=g.nodes(),
        node_color=node_attributes, 
        node_size=100,
        cmap=node_cmap,
        vmin=0,
        vmax=max_na
    )

    ncb = plt.colorbar(nc)
    ncb.set_label(node_str)
    plt.axis('off')
    plt.savefig(os.path.join(directory, file_name))
    plt.clf()
    plt.close()


def draw_x(g, step):
    visit_likelihood = list(nx.get_node_attributes(g,'visitation').values())

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = nx.spring_layout(g, seed=123)
    ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=g.nodes(), node_color=visit_likelihood, 
                                node_size=100, cmap=plt.cm.gist_stern, vmin=0, vmax=1)
    plt.colorbar(nc)
    plt.axis('off')
    plt.savefig('figures/{0:04d}.png'.format(step))
    plt.clf()

def draw_community(g):
    membership = list(nx.get_node_attributes(g,'community').values())
    print(membership)

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = nx.spring_layout(g, seed=123)
    ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=g.nodes(), node_color=membership, 
                                node_size=100, cmap=plt.cm.PiYG, vmin=0, vmax=1)
    plt.colorbar(nc)
    plt.axis('off')
    plt.savefig('figures/community_membership.png')
    plt.clf()


#####
# Below is for plotting interactive html graphs
#####

def draw_graph(g, step):
    _draw_graph(g, show_buttons=True, only_physics_buttons=True)
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get('file:///Users/AndrewDraganov/Coding/graph_visualizations/graph.html')
    driver.set_window_size(1200, 1200)
    driver.get_screenshot_as_file("screenshot_%i.png" % step)
    driver.close()

def _draw_graph(
        g, 
        output_filename='graph.html', 
        show_buttons=False, 
        only_physics_buttons=False
    ):
        """
        This function accepts a networkx graph object,
        converts it to a pyvis network object preserving its node and edge attributes,
        and both returns and saves a dynamic network visualization.

        Valid node attributes include:
            "size", "value", "title", "x", "y", "label", "color".

        Valid edge attributes include:
            "arrowStrikethrough", "hidden", "physics", "title", "value", "width"

        Args:
            g: The graph to convert and display
            output_filename: Where to save the converted network
            show_buttons: Show buttons in saved version of network?
            only_physics_buttons: Show only buttons controlling physics of network?
        """
        # make a pyvis network
        pyvis_graph = net.Network()
        pyvis_graph.width = '700px'
        # for each node and its attributes in the networkx graph
        for node,node_attrs in g.nodes(data=True):
            pyvis_graph.add_node(node,**node_attrs)

        # for each edge and its attributes in the networkx graph
        for source,target,edge_attrs in g.edges(data=True):
            # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
            if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
                # place at key 'value' the weight of the edge
                edge_attrs['value']=edge_attrs['weight']
            # add the edge
            pyvis_graph.add_edge(source,target,**edge_attrs)

        # turn buttons on
        if show_buttons:
            if only_physics_buttons:
                pyvis_graph.show_buttons(filter_=['physics'])
            else:
                pyvis_graph.show_buttons()

        # return and also save
        return pyvis_graph.show(output_filename)

