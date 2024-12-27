import numpy as np
from scipy import linalg as la

class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        A : Column-stochastic transition matrix
        states: A listist of states
        label_to_index: A dictionary mapping each label of the state to the correspoding index of A
    """
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        # Raise ValueError if A is not column stochastic
        if not np.allclose(np.sum(A, axis=0), 1):
            raise ValueError("A is not column stochastic")
        
        n = A.shape[0]

        # Obtain Labels
        if states is None:
            state_labels = list(range(n))
        else:
            state_labels = states

        # Create dictionary
        mydict = {label: i for i, label in enumerate(state_labels)}

        # Store attributes
        self.A = A
        self.states = state_labels
        self.label_to_index = mydict


    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        # Determine column of A that corresponds to provided state label
        current_index = self.label_to_index[state]

        # Get the probabilities of transitioning into each state based on the current state it is in
        transition_probabilities = self.A[:, current_index]

        # Draw from the probabilities
        draw = np.random.multinomial(1, transition_probabilities)

        # Get the index of the draw
        new_index = np.argmax(draw)

        return self.states[new_index]
    
   
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        # Initialize list of N_state_labels and append the start to it
        N_state_labels = []
        N_state_labels.append(start)

        # Transition N - 1 times
        current = start
        for _ in range(N - 1):
            current = self.transition(current)
            N_state_labels.append(current)
        
        return N_state_labels

    
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        # Initialize list of N_state_labels and append the start to it
        N_state_labels = []
        N_state_labels.append(start)

        # Transition until we're at the stop label
        current = start
        while current != stop:
            current = self.transition(current)
            N_state_labels.append(current)
        
        return N_state_labels

    
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        # Generate random state distribution vector x
        n = self.A.shape[0]
        x_current = np.random.rand(n)
        x_current /= np.sum(x_current)

        for _ in range(maxiter+1):
            x_next = self.A @ x_current

            # Check if the 1 norm of the difference between x_k+1 - x_k is less than tolerance at each iteration of k
            if np.linalg.norm(x_next - x_current, ord=1) < tol:
                return x_next
            
            # Update each x_k+1
            x_current = x_next
        
        # If k has exceeded maxiter without finding the norm less than tol, raise ValueError
        raise ValueError("K has exceeded maxiter")


class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        # Read the training set
        with open(filename, 'r') as myfile:
            sentences = myfile.readlines()
        
        # Get the set of unique words in the training set
        unique_words = set()
        for i in sentences:
            words = i.strip().split()
            unique_words.update(words)

        # Add labels
        unique_words.add("$tart")
        unique_words.add("$top")
        state_labels = list(unique_words)

        # Initialize square arrays of zeros to be transition matrix
        n = len(state_labels)
        A = np.zeros((n, n))
        label_to_index = {label: i for i, label in enumerate(state_labels)}

        # Iterate through each sentence in the training set
        for sentence in sentences:
            # Split the sentence into a list of words
            words = sentence.strip().split()

            #Prepend start and stop
            words = ["$tart"] + words + ["$top"]

            # For each consecutive pair of words in list
            for i in range(len(words) - 1):
                x = words[i]
                y = words[i+1]
                current_index = label_to_index[x]
                next_index = label_to_index[y]

                # Add 1 to the correct entry of transition matrix
                A[next_index, current_index] += 1
            
        # Make sure the stop state transitions to itself
        stop_index = label_to_index["$top"]
        A[stop_index, stop_index] = 1

        # Normalize each column      
        col_sums = A.sum(axis=0)
        col_sums[col_sums == 0] = 1 
        A = A / col_sums

        super().__init__(A, state_labels)
    
    # Talk in Yoda 
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        # Create path
        path = self.path("$tart", "$top")

        # Remove the start and stop labels
        words = path[1:-1]

        yapatron = " ".join(words)

        return yapatron
    

if __name__ == "__main__":
    # Write n lines of yoda babble
    y = SentenceGenerator('yoda.txt')
    for _ in range(7):
        print(y.babble())