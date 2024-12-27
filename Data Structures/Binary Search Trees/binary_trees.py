import numpy as np
import time
from matplotlib import pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random as r


class DoublyLinkedListNode:
    """A node with a value and references to the previous and next nodes."""
    def __init__(self, data):
        self.value = data
        self.prev, self.next = None, None

class DoublyLinkedList:
    """A doubly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def __len__(self):
        '''Return the number of nodes in the list.'''
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def __str__(self):
        '''Format and return the list like a standard Python list.'''
        result = []
        current = self.head
        while current:
            result.append(str(current.value))
            current = current.next
        return '[' + ', '.join(result) + ']'

    def insert(self, index, data):
        '''Insert a piece of data as a new node before the given
        index so that the new node is now at index.
        '''
        # Raise index error if index is negative or strictly greater than number of nodes in the list
        if index < 0 or index > len(self):
            raise IndexError
        
        # Initialize new node
        new_node = DoublyLinkedListNode(data)

        # Make a new linked list if we have no nodes yet
        if self.head is None:
            self.head = self.tail = new_node
        # Insert node at the head
        elif index == 0:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        # Insert node at the tail if index is equal to the number of nodes in the list
        elif index == len(self):
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        # Insert node in the middle of the linked list
        else:
            nodes = self.head
            for _ in range(index):
                nodes = nodes.next
            prev_node = nodes.prev
            new_node.prev = prev_node
            new_node.next = nodes
            prev_node.next = new_node
            nodes.prev = new_node
        
        return
        
    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (DoublyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (DoublyLinkedListNode): the node containing the data.
        """
        def find_node(node):
            # Base Case 1: Raise value error if node_to_find is None
            if node is None:
                raise ValueError(str(data) + " is not in the list")
        
            # Base Case 2: node_to_find is our the data we're looking for
            if node.value == data:
                return node
        
            # Recursive case:
            return find_node(node.next)
        
        # Start the recursion by calling the inner function on the head node
        return find_node(self.head)


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        def recursive_find(current_node, data):
            """There's going to be three cases when we're stepping through the tree to find the correct location:
            Case 1: We have found the correct location for the node
            Case 2: We need to go left because data is smaller then current nodes value
            Case 3: We need to go right because data is greater then current nodes value
            Else, this must mean there is already a node in the tree containing the insertion data, so we raise a ValueError
            """
            # Case 1
            if current_node is None:
                return BSTNode(data)
            
            # Case 2
            if data < current_node.value:
                current_node.left = recursive_find(current_node.left, data)
            # Case 3
            elif data > current_node.value:
                current_node.right = recursive_find(current_node.right, data)
            # Else, we need to raise Value Error
            else:
                raise ValueError
            return current_node

        # Assign the root attribute to a new BSTNode containing the data.
        new_node = BSTNode(data)
        if self.root is None:
            self.root = new_node
        else:
            # If we already have a tree, start the recursive process to find and link the parent 
            self.root = recursive_find(self.root, data)

    # Problem 4
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        # Helper function used to find the max valued node
        def find_max(node):
            current = node
            while current.right is not None:
                current = current.right
            return current

        # Helper function to remove the node
        def remove_node(node, key):
            # If we try to remove that doesn't exist, we return None so that the code doesnt break trying to take the value of a node later
            if node is None:
                return None, None
            # Compare the key to the node_value.
            if key < node.value:
                node.left, removed = remove_node(node.left, key)
                if node.left:
                    node.left.prev = node
            elif key > node.value:
                node.right, removed = remove_node(node.right, key)
                if node.right:
                    node.right.prev = node
            else:
                removed = node
                # Case 1: Node is a leaf
                if node.left is None and node.right is None:
                    return None, removed
                
                # Case 2: Node has one child
                elif node.left is None:     # Child is to the right
                    node.right.prev = node.prev
                    return node.right, removed
                elif node.right is None:    # Child is to the left
                    node.left.prev = node.prev
                    return node.left, removed
                
                # Case 4: Node to be removed has two children
                else:
                    # Find the parent, which is the max in left subtree
                    parent = find_max(node.left)
                    # Copy the parent's value to the current node
                    node.value = parent.value
                    # Remove the parent from the left subtree
                    node.left, _ = remove_node(node.left, parent.value)
                    if node.left:  
                        node.left.prev = node

            return node, removed

        # Raise a value error if we're try to remove the node of a nonexistant tree
        if self.root is None:
            raise ValueError("tree doesn't exist brotha")

        # Call our helper function to remove the node and return the new removedd node
        self.root, removed_node = remove_node(self.root, data)
        if self.root:
            self.root.prev = None

        # Raise value eror if tree doesn't contain that node
        if removed_node is None:
            raise ValueError("That node aint in the tree brotha")

        return removed_node
        
    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()

class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self, n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)

def prob5():
    """Compare the build and search times of the DoublyLinkedList, BST, and
    AVL classes. For search times, use DoublyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    with open('english.txt', 'r') as file:
        data = [line.strip() for line in file.readlines()]
    n_values = [2**i for i in range(3, 11)]
    

    # Initialize list to store build times
    dll_times = []
    bst_times = []
    avl_times = []

    # Obtain times for building each data structure
    for n in n_values:
        # Obtain random words
        random_words = r.sample(data, n)

        # Build Doubly Linked List
        dll = DoublyLinkedList()
        start_t = time.time()
        for word in random_words:
            dll.insert(len(dll), word)
        end_t = time.time()
        dll_times.append(end_t - start_t)

        # Build BST
        bst = BST()
        start_t = time.time()
        for word in random_words:
            bst.insert(word)
        end_t = time.time()
        bst_times.append(end_t - start_t)

        # Build AVL
        avl = AVL()
        start_t = time.time()
        for word in random_words:
            avl.insert(word)
        end_t = time.time()
        avl_times.append(end_t - start_t)

    # Initialize lists to store times for searching each data structure
    dll_search_times = []
    bst_search_times = []
    avl_search_times = []

    # Obtain times to search each data structure
    for n in n_values:
        words_to_search = r.sample(random_words, 5)

        # Search Doubly Linked list
        start_t = time.time()
        for word in words_to_search:
            dll.iterative_find(word)
        end_t = time.time()
        dll_search_times.append(end_t - start_t)

        # Search BST
        start_t = time.time()
        for word in words_to_search:
            bst.find(word)
        end_t = time.time()
        bst_search_times.append(end_t - start_t)

        # Search AVL
        start_t = time.time()
        for word in words_to_search:
            avl.find(word)
        end_t = time.time()
        avl_search_times.append(end_t - start_t)

    # Plot of Building Times
    plt.subplot(1, 2, 1)
    plt.plot(n_values, dll_times, label='Doubly Linked List')
    plt.plot(n_values, bst_times, label='BST')
    plt.plot(n_values, avl_times, label='AVL')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N Values')
    plt.ylabel('Build Time')
    plt.title('Times to Build Each Data structure')
    plt.legend()

    # Plot of Searching Times
    plt.subplot(1, 2, 2)
    plt.plot(n_values, dll_search_times, label='Doubly Linked List')
    plt.plot(n_values, bst_search_times, label='BST')
    plt.plot(n_values, avl_search_times, label='AVL')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N Values')
    plt.ylabel('Search Time')
    plt.title('Times To Search Each Data Structure')
    
    plt.tight_layout()
    plt.show()