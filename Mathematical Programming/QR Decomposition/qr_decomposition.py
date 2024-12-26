import numpy as np
from scipy import linalg as la


def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    # Store dimensions of A
    m, n = A.shape
    
    Q = A.copy().astype(float)
    R = np.zeros((n,n))

    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        # Normalize ith column of Q
        Q[:,i] = Q[:,i] / R[i,i]

        for j in range(i+1, n):
            R[i,j] = Q[:,j].T @ Q[:, i]
            # Orthogonalize the jth column of Q
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]

    return Q, R


def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    # Obtain Q and R
    Q, R = la.qr(A)
    # Return absolute value of the decomposition
    decomp = la.norm(la.det(Q)) * la.norm(la.det(R))
    return abs(decomp)
    
# Solve Ax = b system using QR decomposition
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    A = A.astype(float)
    b = b.astype(float)
    
    dim_A = A.shape[0]

    # Compute Q, R
    Q, R = la.qr(A)

    # Compute y =Q^Tb
    y = np.dot(Q.T, b)

    # Use back substitution to solve Rx = y for x
    x = np.zeros(dim_A)
    for k in range(dim_A - 1, -1, -1):
        x[k] = (y[k] - np.dot(R[k, k+1:], x[k+1:])) / R[k, k]

    return x


def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    m, n = A.shape
    R = A.copy().astype(float)
    # The m by m identity matrix
    Q = np.eye(m)
    # Define sign function to be used later
    sign = lambda x: 1 if x >= 0 else -1
    
    for k in range(n):
        u = np.copy(R[k:,k])
        # u0 is the first entry of u
        u[0] = u[0] + sign(u[0])* la.norm(u)
        # Normalize u
        u /= la.norm(u)
        # Apply the reflection to R
        R[k:,k:] -= 2 * np.outer(u,np.dot(u,R[k:,k:]))
        # Apply the reflection to Q
        Q[k:,:] -= 2 * np.outer(u,np.dot(u,Q[k:,:]))
        
    return Q.T, R


def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    m, n = A.shape
    # Initialize matrices
    H = A.copy().astype(float)
    Q = np.eye(m)

    #Initialize sign function
    sign = lambda x: 1 if x >= 0 else -1

    for k in range(n-2):
        u = np.copy(H[k+1:,k])
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u /= la.norm(u)
        
        # Apply Qk to H
        H[k+1:, k:] -= 2 * np.outer(u, np.dot(u.T, H[k+1:, k:]))
        # # Apply Qk^T to H
        H[:, k+1:] -= 2 * np.outer(np.dot(H[:, k+1:], u), u.T)
        # # Apply Qk to Q
        Q[k+1:, :] -= 2 * np.outer(u, np.dot(u.T, Q[k+1:, :]))

    return H, Q.T