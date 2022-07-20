import random
import numpy as np
import matplotlib.pyplot as plt


class Graph:
    """
    Undirected graph without loops
    """
    def __init__(self, graph=None):
        if graph is None:
            graph = {}
        self.graph = graph

    def __len__(self):
        return len(self.graph)

    def edges(self):
        return [(node, neighbor)
                for node in self.graph
                for neighbor in self.graph[node] if node < neighbor]

    def nodes(self):
        return list(self.graph.keys())

    def nice_print(self):
        return '\n'.join([f'{node}: {self.graph[node]}' for node in self.nodes()])

    def add_node(self, node: int):
        if node not in self.graph:
            self.graph[node] = set()

    def add_edge(self, node1: int, node2: int):
        if node1 not in self.graph:
            self.add_node(node1)
        if node2 not in self.graph:
            self.add_node(node2)
        if node1 != node2:
            self.graph[node1].add(node2)
            self.graph[node2].add(node1)

    def generate_random_graph(
        self,
        n_nodes: int = None,
        max_nodes: int = 20,
        p: float = 0.5
    ) -> None:
        """
        Generates random graph. By default with random number of nodes
        n_nodes: exact number of nodes to generate
        max_nodes: maximum number of nodes that could be generated
        p: probability of generation each edge
        """
        if n_nodes is None:
            n_nodes = random.randint(1, max_nodes)

        for node in range(n_nodes):
            self.add_node(node)

        for node1 in range(n_nodes):
            for node2 in range(node1 + 1, n_nodes):
                if random.random() < p:
                    self.add_edge(node1, node2)

    def plot_graph(self, nodes_subset: set = None, path: str = None) -> None:
        """
        Plots the graph on in a circle layout
        nodes_subset: nodes, that would be highlighted with red
        path: path to save image
        """
        theta = np.linspace(0, 1, len(self.graph) + 1)[:-1] * 2 * np.pi
        xs = np.cos(theta)
        ys = np.sin(theta)

        plt.figure(figsize=(10, 10))
        for (node1, node2) in self.edges():
            x1 = xs[node1]
            x2 = xs[node2]
            y1 = ys[node1]
            y2 = ys[node2]
            plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2, zorder=1)

        plt.scatter(xs, ys, color='b', zorder=2)
        if nodes_subset:
            for node in nodes_subset:
                plt.scatter(xs[node], ys[node], color='r', s=70, zorder=3)

        for node in self.nodes():
            plt.text(xs[node] * 1.1, ys[node] * 1.1, str(node), size='xx-large')
        plt.axis('off')

        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()


def is_independent(G: Graph, curr_set: set) -> bool:
    """
    Function checks if curr_set is an independent set in graph G
    """
    for node in curr_set:
        if G.graph[node] & curr_set:
            return False
    return True


def max_independent_set(G: Graph) -> set:
    """
    Function returns the maximum independent set in the graph G
    """
    n = len(G)
    form = f'0{n}b'

    max_num = 0
    max_set = set()
    for i in range(1, 2 ** n):
        binary_mask = format(i, form)

        curr_set = set()
        for j in range(n):
            if binary_mask[j] == '1':
                curr_set.add(j)

        if is_independent(G, curr_set) and len(curr_set) > max_num:
            max_num = len(curr_set)
            max_set = curr_set.copy()

    return max_set


def main():
    for n_nodes in [4, 5, 8, 11, 15, 17]:
        G = Graph()
        G.generate_random_graph(n_nodes=n_nodes, p=0.4)

        friends = max_independent_set(G)

        with open(f'algorithm_result/graph_{n_nodes}.txt', 'w') as output:
            output.write(G.nice_print() + f'\n\nResult: {friends}')

        G.plot_graph(nodes_subset=friends, path=f'algorithm_result/{n_nodes}.png')


if __name__ == '__main__':
    main()
