import numpy as np
from scipy import linalg as la


def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    # Compute D
    D = np.diag(A.sum(axis=1))

    # Use the Laplace Equation to get L
    L = D - A

    return L

def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    # Get the Laplacian of the graph G
    L = laplacian(A)

    # Obtain the eigenvalues of the Laplacian
    evals = la.eigvals(L)
    real_evals = np.real(evals)

    # Get # of components essentially equal to 0, which is the # of connected components
    connected_components_num = 0
    for eval in real_evals:
        if eval < tol:
            connected_components_num += 1
    
    # Obtain algebraic connectivity
    sorted = np.sort(real_evals)
    algebraic_connectedness = sorted[1]        
    
    return connected_components_num, algebraic_connectedness