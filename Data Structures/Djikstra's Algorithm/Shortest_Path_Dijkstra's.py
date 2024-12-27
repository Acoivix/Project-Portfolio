from math import sqrt
from queue import PriorityQueue
from collections import defaultdict

class Edge:
    """An edge object, which wraps the node and weight attributes into one
    object, allowing for insertion/deletion from a set using just
    the node attribute

    Attributes:
        node (str): the value for the node the edge is pointing to
        weight (int): the weight of the edge
    """
    def __init__(self, node, weight):
        self.node = node
        self.weight = weight

    def __hash__(self):
        """Use only node attribute for hashing"""
        return hash(self.node)

    def __eq__(self, other):
        """Use only node attribute for equality"""
        if isinstance(other, Edge):
            return self.node == other.node
        return self.node == other

    def __str__(self):
        """String representation: a tuple-like view of the node and weight"""
        return f"({str(self.node)}, {str(self.weight)})"

    def __repr__(self):
        """Repr is used when edges are displayed in a set"""
        return f"Edge({repr(self.node)}, {repr(self.weight)})"

class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors (stored as tuples).

    Attributes:
        d (dict): the adjacency dictionary of the graph.
        directed (bool): true if the graph is a directed graph.
    """
    def __init__(self, adjacency={}, directed=False):
        """Store the adjacency dictionary and directed as class attributes"""
        # Store and save attribute self.d and self.directed
        self.d = adjacency
        self.directed = directed    

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        # Add node if not in graph
        if n not in self.d:
            self.d[n] = set()

    def add_edge(self, u, v, weight=1.0):
        """Add a weighted edge between node u and node v.
        If an edge already exists between u and v, simply update the weight.
        Also add u and v to the graph if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        # Add nodes u,v if not already present in the graph
        if u not in self.d:
            self.d[u] = set()
        if v not in self.d:
            self.d[v] = set()
        
        edge_to_add = Edge(v, weight)

        # Update weight of edge if edge already present.
        if edge_to_add in self.d:
            self.d[u].discard(edge_to_add)               
        self.d[u].add(edge_to_add)

        # If the graph is undirected do the reverse process aswell
        if not self.directed:
            edge_v_to_u = Edge(u, weight)
            if edge_v_to_u in self.d:
                self.d[v].discard(edge_v_to_u)
            self.d[v].add(edge_v_to_u)

    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        # Raise KeyError if node not in graph
        if n not in self.d:
            raise KeyError(f"Node {n} not in graph")
        
        edge_to_remove = Edge(n, 0)

        # Remove edges by first checking case of undirected graph
        if not self.directed:
            for neighbor in self.d[n]:
                self.d[neighbor.node].discard(edge_to_remove)
        # If graph is directed
        else:
            for neighbor in self.d.values():
                neighbor.discard(edge_to_remove)

        # Remove node
        del self.d[n]

    def remove_edge(self, u, v):
        """Remove the edge starting at u and ending at v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge from u to v.
        """
        # Raise KeyError if either node is not in graph
        if u not in self.d:
            raise KeyError(f"Node {u} not in graph")
        if v not in self.d:
            raise KeyError(f"Node {v} not in graph")
        
        # Raise KeyError if there is no edge between nodes
        edge_to_remove = Edge(v, 0)
        if edge_to_remove not in self.d[u]:
            raise KeyError(f"No edge exists between {u} and {v}")

        # Remove the edge
        self.d[u].discard(edge_to_remove)

        # If the graph is undirected, remove the reverse edge aswell
        if not self.directed:
            reverse_edge_to_remove = Edge(u, 0)
            if reverse_edge_to_remove not in self.d[v]:
                raise KeyError(f"No edge exists between {v} and {u}")
            self.d[v].remove(reverse_edge_to_remove)


    def shortest_path(self, source, target):
        """Begin Dijkstra's at the source node and proceed until the target is
        found. Return an integer denoting the sum of weights along the shortest
        path from source to target along with a list of the path itself,
        including endpoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            An int denoting the sum of weights along the shortest path
                from source to target
            A list of nodes along the shortest path from source to target,
                including the endpoints. The path should contain strings
                representing the nodes, not edge objects

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        # Raise a KeyError if the input nodes are not in the graph
        if source not in self.d:
            raise KeyError(f"{source} not in graph")
        if target not in self.d:
            raise KeyError(f"{target} not in graph")
        
        # Initialize needed data structures
        Q = PriorityQueue()
        V = set()                                       # Processed nodes
        d = {node: float('inf') for node in self.d}     # Shortest distance to each node. They all start at infinity in Dijkstra's Algorithm
        p = {node: None for node in self.d}             # Predecessors dictionary

        # Process the source node
        Q.put((0, source))
        d[source] = 0

        while not Q.empty():
            # Pop the current node off Q
            current_distance, current_node = Q.get()

            # If current node is the destination, finish the loop
            if current_node == target:
                break   

            # Visit the current node
            V.add(current_node)

            # Loop through all the neighbors of current.    
            for edge in self.d[current_node]:
                neighbor = edge.node
                weight = edge.weight
                # Obtain potential new shorter distance
                new_distance = current_distance + weight
                
                # Update distances if we have found shorter distance
                if new_distance < d[neighbor]:
                    d[neighbor] = new_distance
                    
                    # Process predecessors and Q 
                    p[neighbor] = current_node
                    Q.put((new_distance, neighbor))

        # Reconstruct the optimal path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = p[current]
            
        path.reverse()

        return d[target], path