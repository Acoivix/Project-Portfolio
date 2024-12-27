import numpy as np
from scipy import stats
from scipy import linalg as la
from scipy.spatial import KDTree


def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    # Calculate distance between X and z
    distance = la.norm(X-z, axis = 1)

    # Locate index of min distances
    index_min = np.argmin(distance)

    # Calculate and return x* and d*
    return X[index_min], distance[index_min]

# Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        # Raise Typeerror if not of type np.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a NumPy array')
        
        # Else, set left, right, values, and pivot accordingly
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None
        
# KD Tree Class
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Insert new node
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        # If the tree is empty
        if self.root is None:
            self.root = KDTNode(data)
            self.root.pivot = 0
            self.k = len(data)
            return
        
        # Raise ValueError if data to be inserted is not in k
        if len(data) != self.k:
            raise ValueError("Data to be inserted is not in R^k")
        
        # Helper function to traverse the tree recursively to find where to insert the node
        def recursive_traverse(current):
            """Returns a tuple of (parent_node, is_left_child) where is_left_child indicates if we go left"""
            # Raise ValueError if there is already a node in the tree containing x,
            if np.allclose(data, current.value):
                raise ValueError("There is already a node in the tree containing this data")
            
            # If we're going left
            if data[current.pivot] < current.value[current.pivot]:
                if current.left is None:   # determine if there is something to the left or not yet
                    new_node = KDTNode(data)    # Create new node if there is nothing yet
                    
                    # Increment parent's pivot if needed
                    if current.pivot < self.k - 1:
                        new_node.pivot = current.pivot + 1
                    else:
                        new_node.pivot = 0
                    
                    current.left = new_node 
                # If we already have node there, traverse to the next node
                else:
                    recursive_traverse(current.left)
            # If we are going right
            else: 
                # Use similar logic to construct the case of going to the right
                if current.right is None:
                    new_node = KDTNode(data)
                    
                    # Increment parent's pivot if needed
                    if current.pivot < self.k - 1:
                        new_node.pivot = current.pivot + 1
                    else:
                        new_node.pivot = 0

                    current.right = new_node
                else:
                    recursive_traverse(current.right)
            
        # Start recursion at the root
        recursive_traverse(self.root)

    # Search KDT
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        # Implement distance helper funcion to be used in algorithm 1
        def distance(x, y):
            return np.linalg.norm(x - y)

        def KD_Nearest_Neighbor_Search(current, nearest, d_asterick):
            if current is None:
                return nearest, d_asterick

            x = current.value
            i = current.pivot

            # Check if current is closer to z than nearest
            current_distance = distance(x, z)
            if current_distance < d_asterick:
                nearest = current
                d_asterick = current_distance
            
            # Search to the left 
            if z[i] < x[i]:
                nearest, d_asterick = KD_Nearest_Neighbor_Search(current.left, nearest, d_asterick)
                if z[i] + d_asterick >= x[i]:
                    nearest, d_asterick = KD_Nearest_Neighbor_Search(current.right, nearest, d_asterick)
            # Or, search to the right
            else:
                nearest, d_asterick = KD_Nearest_Neighbor_Search(current.right, nearest, d_asterick)
                if z[i] - d_asterick <= x[i]:
                    nearest, d_asterick = KD_Nearest_Neighbor_Search(current.left, nearest, d_asterick)
            return nearest, d_asterick
        
        nearest, d_asterick = KD_Nearest_Neighbor_Search(self.root,self.root, distance(self.root.value, z))
        
        return nearest.value, d_asterick
                
    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)
    

# Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        """Initiate classifier
        Attributes: n_neighbors (int)
        """
        self.n_neighbors = n_neighbors
        self.tree = None
        self.labels = None

    def fit(self, X, y):
        """Save the tree and lables as attributes"""
        # Assure that arrays are correct dimensionally
        if len(X.shape) != 2:
            raise ValueError("X must be a 2-dimensional array")
        if len(y.shape) != 1:
            raise ValueError("y must be a 1-dimensional array")

        # Save the tree and lables as attributes
        self.labels = y
        self.tree = KDTree(X)
    
    def predict(self, z):
        """Predict the label for z"""
        # Assure that z is one dimensional
        if z.shape != (self.tree.data.shape[1],):
            raise ValueError("z must be one dimension")
        
        # Query the tree for k nearest neigbhors
        distances, indices = self.tree.query(z,k = self.n_neighbors)

        neighbor_labels = self.labels[indices]

        # most_common_label = stats.mode(neighbor_labels)[0][0]
        # Use alternate method to split tie and obtain most common label since my computer's broken version of python cant use stats.mode() correctly
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]

        return most_common_label
    

def train(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    # Load the MNIST subset data
    data = np.load(filename)
    X_train = data["X_train"].astype(np.float64) # Training data
    y_train = data["y_train"] # Training labels
    X_test = data["X_test"].astype(np.float64) # Test data
    y_test = data["y_test"] # Test labels

    knc = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the classifier and training data
    knc.fit(X_train, y_train)
    
    # Obtain our predictions of labels
    predictions = [knc.predict(X_test[i]) for i in range(len(X_test))]

    # Calculate accuracy of label prediction
    accuracy = np.mean(predictions == y_test)

    return accuracy 

    