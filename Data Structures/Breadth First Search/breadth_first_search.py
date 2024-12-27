import networkx as nx
from numpy import mean
from collections import deque, OrderedDict
from matplotlib import pyplot as plt


class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""

        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        # Initialize dictionary if no dictionary exists yet
        if n not in self.d:
            self.d[n] = set()

    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        # Add nodes to graph if not present
        if u not in self.d:
            self.add_node(u)
        if v not in self.d:
            self.add_node(v)
        # Add edge between nodes
        self.d[u].add(v)
        self.d[v].add(u)

    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        # Remove node; pop raises KeyError if node not in d
        if n not in self.d:
            raise KeyError("Node not in graph")
        
        # Delete node n
        del self.d[n]

        # Remove n from all neighbor sets
        for i in self.d[n]:
            self.d[i].remove(n)
        
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        # Raise KeyError if node not in graph
        if u not in self.d or v not in self.d:
            raise KeyError("One of these nodes is not in graph")
        # Raise KeyError if there is no edges between nodes
        if v not in self.d[u] or u not in self.d[v]:
            raise KeyError("No edge between nodes")
        # Remove edge between nodes
        self.d[u].remove(v)
        self.d[v].remove(u)

    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        print(self.d)
        # Raise KeyError if source node not in graph
        if source not in self.d:
            raise KeyError("Crazy")
        
        # Initialize Visited Nodes, Queue, Marked Nodes respectively
        V = []
        Q = deque(source)
        M = set(source)

        # Traversal is done when Q is empty
        while len(Q) > 0:
            # Pop node off of Q
            current = Q.popleft()         
            # Visit current node
            V.append(current)
            
            # Add neigbors of current node that are not in M to Q and M
            for l in self.d[current]:
                # print(l)
                if l not in M:
                    Q.append(l)
                    M.add(l)
            # Q.extend(neighbor for neighbor in self.d[current] if neighbor not in M)
            # M.update(neighbor for neighbor in self.d[current])

        return V

    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        # Raise KeyError ff either of the input nodes are not in the graph
        if source not in self.d or target not in self.d:
            raise KeyError("Nodes must be in the graph")
        
        # Initialize visited nodes,  Q, and predecessor dictionary
        V = set()
        Q = deque([source])
        predecessor = {}
        
        # Return the node if the target node is the source node
        if source == target:
            return [source]
        
        # Else, we traverse by checking neigboring nodes to find the target
        while len(Q) > 0:
            current = Q.popleft()                    # Get current node
            for neighbor in self.d[current]:         # Traverse through each neigbhor
                if neighbor not in V:                # Check if neighbor hasn't been visited yet
                    V.add(neighbor)
                    Q.append(neighbor)
                    # Map the predecessor of our node to the current node
                    predecessor[neighbor] = current

                    # Check to see if each neighbor is the target
                    if neighbor == target:
                        shortest_path = []
                        shortest_path.append(target)
                        
                        # Append each predecessor in reverse order to obtain the shortest path
                        while shortest_path[-1] != source:
                            shortest_path.append(predecessor[shortest_path[-1]])
                        
                        shortest_path.reverse()
                        return shortest_path

        # Return none if no path exists
        return None


class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        # Initialize title, actor, and nx_graph
        self.titles = set()
        self.actors = set()
        self.graph = nx.Graph()

        # Open file 
        with open(filename, "r") as data:
            # Parse through each line
            for line in data:
                movie_stats = line.strip().split("/")

                # Add the title to the set of movies 
                title = movie_stats[0]
                self.titles.add(title)
                # Add the cast members to the set of actors
                self.graph.add_node(title)

                # Add an edge to the graph between the movie and each cast member
                cast = movie_stats[1:]
                for actor in cast:
                    self.actors.add(actor)
                    self.graph.add_node(actor)
                    self.graph.add_edge(title, actor)

    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        # Obtain shortest path
        shortest_path = nx.shortest_path(self.graph, source, target)

        # Obtain degree of seperation
        degree_of_seperation = (len(shortest_path) - 1) // 2            # We divide by 2 to account for the fact that the tile shouldn't effect length of degree of seperation

        return shortest_path, degree_of_seperation


    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        # Get all shortest path lengths
        shortest_path_lengths = nx.single_source_shortest_path_length(self.graph, target) 
        
        # Divide length by two to exclude movie titles in the path length
        actor_only_length = [(length) // 2 for node, length in shortest_path_lengths.items() if node in self.actors and node != target]

        # Obtain average path length
        avg_path_lengths = sum(actor_only_length) / len(actor_only_length)
       
        # Plot the distribution of path lengths
        plt.hist(actor_only_length, bins=[i-.5 for i in range(8)])
        plt.title(f"Degree of Seperation in movies to {target}")
        plt.xlabel('Degrees of Separation')
        plt.ylabel('Number of Actors')
        plt.grid(True)
        plt.show()

        return avg_path_lengths                 # Return average path lengths
