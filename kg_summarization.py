import numpy as np
import os
import math
import networkx as nx
from visualizer import draw_graph, draw_full_graph, draw_subgraph
import matplotlib.pyplot as plt
from utils import symmetrize
import sys
from PIL import Image
np.set_printoptions(threshold=np.inf)

class Walker:
    def __init__(self, loc, ID):
        self.loc = loc
        self.ID = ID

class Relation:
    def __init__(self, x_loc, y_loc, rel_type):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.rel_type = rel_type
        self.walks = {}
        self.total_walks = 0
        self.most_walked = False

    def get_total_walks(self):
        self.total_walks = 0
        for _, walks in self.walks.items():
            self.total_walks += walks

    def add_walk(self, walker):
        if walker.ID in self.walks:
            self.walks[walker.ID] += 1
        else:
            self.walks[walker.ID] = 1
        self.get_total_walks()

    def remove_walker(self, walker):
        if walker.ID in self.walks:
            self.walks.pop(walker.ID)
        self.get_total_walks()

class KGSummarizer:
    """
    Code to make a simulated knowledge graph and summarize the queried information
    by applying random walkers to capture information convergence likelihoods
        - The knowledge graph receives queries at every time-step on a random node
        - Random walkers are spawned each time a 'query' is made and each represents
          one sub-unit of relevance for the query
        - At every time-step, each walker takes a step along a random edge
          and a random set of c walkers is removed
        - The queries are summarized by the nodes with the highest walker densities
        - If a node with high degree has been queried, we are likely to have
          a large collection of walkers remain on the node or in its vicinity
            - If, instead, a node with small degree is queried, the walkers will
              converge to neighboring nodes with high degree
    """
    def __init__(
        self,
        n=50,
        K=30,
        c=10,
        num_queries=40
    ):
        self.n = n
        self.K = K
        self.c = c
        self.num_queries = num_queries
        self.walker_id = 0
        self.num_walkers = 3 * self.n
        self.fullgraph_name = 'full_graph_{0:04d}.png'
        self.subgraph_name = 'subgraph_{0:04d}.png'
        self.directory = 'KG_figures'

        self.create_random_graph()
        self.create_random_queries()
        self.init_relations()
        self.init_walkers()

    def create_random_graph(self):
        adj_mat = np.zeros([self.n, self.n])
        # Create an adjacency matrix with some kind of non-uniform structure
        for i in range(self.n):
            for j in range(i):
                adj_mat[i, j] = math.sin(((i+1)*(j+1))**2 / (20 * math.pi))

        # threshold the adj_mat
        thresh = 0.95
        adj_mat[adj_mat > thresh] = 1
        adj_mat[adj_mat <= thresh] = 0
        np.fill_diagonal(adj_mat, 1)

        # Fill edge types matrix with values between 1-5
        #   then mask by adjacency matrix of 0s and 1s
        num_relations = 5
        relation_consistency_threshold = 0.5

        # Def: Relations are 'random' if their type is independent of the node they
        #      are attached to
        random_relations = np.random.rand(self.n, self.n) * num_relations + 2
        random_relations *= adj_mat
        random_relations = random_relations.astype(np.int32)

        # Def: Relations are 'consistent' if, for node N, all relations on N
        #      are of the same type
        consistent_relations = np.stack([np.arange(self.n) for i in range(self.n)])
        consistent_relations = consistent_relations % num_relations + 2
        consistent_relations *= adj_mat.astype(np.int32)

        # We want relations that are somewhat consistent for their respective nodes
        mask = np.random.rand(self.n, self.n) < relation_consistency_threshold
        mask = mask.astype(np.int32)
        edge_types = random_relations * mask + consistent_relations * (1 - mask)
        # relation_type 1 is reserved for identity relationships (self-loops)
        np.fill_diagonal(edge_types, 1)

        # Symmetrize matrices
        self.adj_mat = symmetrize(adj_mat, self.n)
        self.edge_types = symmetrize(edge_types, self.n)
        self.degree_vector = np.sum(adj_mat, axis=1) / 2
        self.sum_degrees = np.sum(self.degree_vector)

        empty_diag = self.adj_mat.copy()
        np.fill_diagonal(empty_diag, 0)
        self.g = nx.convert_matrix.from_numpy_matrix(empty_diag)

    def create_random_queries(self):
        self.queries = np.random.randint(0, self.n, self.num_queries)

    def init_relations(self):
        self.relations = {}
        for i in range(self.n):
            for j in range(self.n):
                if self.adj_mat[i, j] > 0:
                    self.relations[(i, j)] = Relation(i, j, self.edge_types[i, j])

    def append_walker(self, array, loc):
        array.append(Walker(loc, self.walker_id))
        self.walker_id += 1
        return array

    def init_walkers(self):
        # num_walkers is rough estimate on actual number of walkers (due to float rounding)
        walker_allocations = (self.degree_vector * self.num_walkers / self.sum_degrees)
        walker_allocations = np.round(walker_allocations).astype(np.int32)
        self.walkers = []
        for i in range(self.n):
            for j in range(walker_allocations[i]):
                self.append_walker(self.walkers, i)

    def process_queries(self):
        for t, query in enumerate(self.queries):
            self.remove_walkers()
            self.initialize_new_walkers(query)
            self.step_walkers()
            locs = self.determine_subgraph()
            f1_score = self.get_f1_score(locs, t)

            self.plot_full_graph(t)
            self.plot_subgraph(locs, t)

    def remove_walkers(self):
        w = len(self.walkers)
        keep_indices = np.random.choice(np.arange(w), w - self.c, replace=False)
        remove_indices = np.array([i for i in range(w) if i not in keep_indices])
        for i in remove_indices:
            dead_walker = self.walkers[i]
            for _, relation in self.relations.items():
                relation.remove_walker(dead_walker)
        live_walkers = []
        for i in keep_indices:
            live_walkers.append(self.walkers[i])
        self.walkers = live_walkers

    def initialize_new_walkers(self, query):
        new_walkers = []
        for i in range(self.c):
            self.append_walker(new_walkers, query)
        self.walkers = self.walkers + new_walkers

    def step_walkers(self):
        for i, walker in enumerate(self.walkers):
            current_node = self.walkers[i].loc
            next_nodes = np.unique(np.where(self.adj_mat[current_node] == 1))
            next_node = np.random.choice(next_nodes)
            self.relations[(current_node, next_node)].add_walk(walker)
            self.walkers[i].loc = next_node

    def determine_subgraph(self):
        subgraph_adj = np.zeros([self.n, self.n])
        total_walks = np.zeros(len(self.relations))
        locs = np.zeros((len(self.relations), 2))
        for i, (loc, relation) in enumerate(self.relations.items()):
            total_walks[i] = relation.total_walks
            locs[i] = [loc[0], loc[1]]
        relations_sort = np.argsort(-total_walks)
        return locs[relations_sort[:int(self.K/2)]]

    def get_f1_score(self, locs, t):
        # True positives
        tp = 0
        fps = []
        fn = 0
        relevant_queries = self.queries[max(0, t-20):t]
        for node in relevant_queries:
            if any([node in k for k in locs]):
                tp += 1
            else:
                fn += 1
        for loc in locs:
            for i in loc:
                if i not in relevant_queries:
                    fps.append(i)
        fp = len(np.unique(np.array(fps)))
        print(tp, fp, fn)
        print(tp / (tp + 0.5 * (fn + fp)))


    def plot_full_graph(self, t):
        """
        In plotting the full graph we want:
            - edge transparency as number of walks
            - node coloring by number of queries on it
        """
        past_queries = self.queries[max(0, t-20):t]
        edge_attributes = {key: self.relations[key].total_walks for key in self.relations}
        nx.set_edge_attributes(self.g, edge_attributes, 'num walks along edge')

        node_visits = {i: 0 for i in range(self.n)}
        for key in self.relations:
            node_visits[key[1]] += 1
        nx.set_node_attributes(self.g, node_visits, 'visits to node')

        node_queries = {}
        for node in range(self.n):
            if node in past_queries:
                node_queries[node] = 1
            else:
                node_queries[node] = 0
        nx.set_node_attributes(self.g, node_queries, 'queried')

        draw_full_graph(
            self.g,
            node_str='visits to node',
            edge_str='num walks along edge',
            queried_str='queried',
            node_cmap=plt.cm.Blues,
            edge_cmap=plt.cm.YlGn,
            file_name=self.fullgraph_name.format(t),
            directory=self.directory
        )

    def plot_subgraph(self, locs, t):
        """
        In plotting the subgraph we want:
            - edge binary transparency by subgraph adjacency matrix
            - node binary transparency by subgraph adjacency matrix

        Want graph layout to look identical to full graph for easy comparison
        """
        edge_attributes = {k: 0 for k in self.relations}
        node_attributes = {i: 0 for i in range(self.n)}
        count = 0
        for loc in locs:
            node_attributes[int(loc[0])] = 1
            node_attributes[int(loc[1])] = 1
            edge_attributes[tuple(loc)] = 1
            edge_attributes[tuple(loc[::-1])] = 1

        nx.set_node_attributes(self.g, node_attributes, 'subgraph_node')
        nx.set_edge_attributes(self.g, edge_attributes, 'subgraph_edge')
        draw_subgraph(
            self.g,
            node_str='subgraph_node',
            edge_str='subgraph_edge',
            node_cmap=plt.cm.binary,
            edge_cmap=plt.cm.binary,
            file_name=self.subgraph_name.format(t),
            directory=self.directory
        )

    def combine_images(self):
        for t in range(len(self.queries)):
            image_paths = [self.fullgraph_name.format(t), self.subgraph_name.format(t)]
            image_paths = [os.path.join(self.directory, p) for p in image_paths]
            images = [Image.open(x) for x in image_paths]
            widths, heights = zip(*(i.size for i in images))

            total_height = sum(heights)
            max_width = max(widths)
            new_im = Image.new('RGB', (max_width, total_height))

            y_offset = 0
            for im in images:
              new_im.paste(im, (0, y_offset))
              y_offset += im.size[1]

            new_im.save(os.path.join(self.directory, 'combined_{0:04d}.png'.format(t)))

    def clear_single_images(self):
        for t in range(len(self.queries)):
            image_paths = [self.fullgraph_name.format(t), self.subgraph_name.format(t)]
            image_paths = [os.path.join(self.directory, p) for p in image_paths]
            for path in image_paths:
                os.remove(path)

def save(num_frames, video_length):
    frame_rate = num_frames / float(video_length)
    os.system("ffmpeg -y -framerate {:.4f} -pattern_type glob -i 'KG_figures/combined_*.png' -c:v libx264 -r 30 -pix_fmt yuv420p movie.mp4".format(frame_rate))

if __name__ == '__main__':
    kg = KGSummarizer()
    kg.process_queries()
    print('combining images...')
    kg.combine_images()
    kg.clear_single_images()
    save(len(kg.queries), 20)
