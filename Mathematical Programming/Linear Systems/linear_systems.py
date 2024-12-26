import numpy as np
from time import perf_counter as time
from scipy import sparse
from scipy import linalg as la
from scipy.sparse import linalg as spla
from matplotlib import pyplot as plt


def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    dim_A = A.shape[0]
   
    A = A.astype(float)
    
    for i in range(dim_A):
        diag_element = A[i,i]
        
        #Eliminate terms under the leading entry
        for pivot_row in range(i+1, dim_A):
            tmp = A[pivot_row,i] / diag_element       #get terms under the leading entry
            A[pivot_row] = A[pivot_row] - tmp * A[i]

    return A
    
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    A = A.astype(float)
    
    # Obtain dimension of A and initialize L and U accordingly
    dim_A = A.shape[0]
    U = np.copy(A)
    L = np.eye(dim_A)

    # Iterate through each column
    for col in range(dim_A):            
        for row in range(col + 1, dim_A):
            tmp = U[row, col] / U[col, col]     # Define tmp variable that fills L and updates U
            L[row, col] = tmp                   # Build L   
            U[row, :] -= tmp * U[col, :]  # Build U
    
    return L, U

def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((n,) ndarray): The solution to the linear system.
    """
    A = A.astype(float)
    b = b.astype(float)
    # Get the LU Decomposition
    L, U = lu(A)
    
    dim_A = A.shape[0]

    # Build Y to be used for forward substitution
    y = np.zeros(dim_A)

    # Solve for Ly = Pb. Since P = I its just Ly = b
    for k in range(dim_A):
        y[k] = b[k] - np.dot(L[k, :k], y[:k])

    #Build x to be solved for using backwards substitution substitution
    x = np.zeros(dim_A)  

    #Solve for Ux = y
    for k in range(dim_A - 1, -1, -1):
        x[k] = (1 / U[k,k])*(y[k] - np.dot(U[k, k+1:], x[k+1:]))

    return x

def solving_linear_systems_timing():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    n_values = [10, 20, 50, 100, 200]

    # Initialize lists to store times
    inv_times = []
    solve_times = []
    lu_full_times = []
    lu_times = []

    for n in n_values:
        # Generate random matrices A and b of n by n and n size respectively
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        
        # Obtain times for method: Invert A and left multiply inverse to b
        time_s = time()
        A_inverse = la.inv(A)
        _ = A_inverse@b
        inv_times.append(time() - time_s)

        # Obtain times for method: la.solve()
        time_s = time()
        _ = la.solve(A,b)
        solve_times.append(time() - time_s)

        # Obtain times for method: Use la.lu_factor() and solve with LU decomposition
        time_s = time()
        lu, piv = la.lu_factor(A)
        _ = la.lu_solve((lu, piv), b)
        lu_full_times.append(time() - time_s)

        # Obtain times for method: LU decomposition not including time to get factor
        lu, piv = la.lu_factor(A)
        time_s = time()
        _ = la.lu_solve((lu, piv), b)
        lu_times.append(time() - time_s)
    
# Plot results with n_values versus times
    plt.plot(n_values, inv_times, label = "Inverse method")
    plt.plot(n_values, solve_times, label = "Solve method")
    plt.plot(n_values, lu_full_times, label = "Factor then solve using LU decomposition")
    plt.plot(n_values, lu_times, label = "Lu decomposition not including time to factor")

    plt.title("Time to solve for Ax=b using different methods!")
    plt.xlabel("N values of matrix size")
    plt.ylabel('Time to compute')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

# Problem 5
def sparce_matrices(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    I = np.eye(n)
    
    # Construct our B matrix by setting diagonals and offsets
    diagonals = ([1 for i in range(n-1)], [-4 for i in range(n)], [1 for i in range(n)])
    offsets = [-1,0,1]
    B = sparse.diags(diagonals, offsets, shape=(n, n))
    
    # Construct A. Start with initializing blocks
    blocks = []
    
    for i in range(n):
        ind_row_blocks = []             # Initialize individual row blocks to append each row
        for j in range(n):
            if i == j:
                ind_row_blocks.append(B)
            elif i == j+1 or i == j-1:
                ind_row_blocks.append(I)
            else:
                ind_row_blocks.append(None)
        # Appened our individual row blocks to the main blocks
        blocks.append(ind_row_blocks)        

    # Create the array out of our bloacks
    A = sparse.bmat(blocks, format = 'bsr')
    
    return A
    
# Problem 6
def timing_and_plotting():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    n_values = [i for i in range(1,10)]

    # Initialize list to store times for sparse solve and regular solve
    sparse_solve_times = []
    reg_solve_times = []

    for n in n_values:
        # Generate A matrix and random n^2 b vector
        A = sparce_matrices(n)
        b = np.random.rand(n**2)
        
        # Time sparse solve (exluding conversion)
        A_csr = A.tocsr()  # Convert to CSR
        time_s = time()
        _ = spla.spsolve(A_csr, b)
        sparse_solve_times.append(time() - time_s)

        # Time regular solve (excluding conversion)
        A_reg = A.toarray()
        time_s = time()
        _ = la.solve(A_reg, b)
        reg_solve_times.append(time() - time_s)
    
    n_squared_values = [n**2 for n in n_values]                 # Convert n_values to n_squared values for plotting

    # Plot n^2 values versus execution times for sparse solve and regular solve
    plt.plot(n_squared_values, sparse_solve_times, label = "Sparse solve times")
    plt.plot(n_squared_values, reg_solve_times, label = "Regular solve times")
    plt.xlabel("n^2 values")
    plt.ylabel("Execution times for each method")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
