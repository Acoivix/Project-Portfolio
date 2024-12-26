import time
import numpy as np
from random import random
from matplotlib import pyplot as plt


def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    
    # Construct Matrix Representation
    starting_matrix = np.array([[a,0],[0,b]])
    
    return np.matmul(starting_matrix, A)
    

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Construct Matrix Representation
    starting_matrix = np.array([[1,a],[b,1]])

    return np.matmul(starting_matrix, A)

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Construct Matrix Representation
    starting_matrix = (1/(a**2 + b**2))* np.array([[a**2 - b**2, 2*a*b], [2*a*b, b**2 - a**2]])

    return np.matmul(starting_matrix, A)

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Construct Matrix Representation
    starting_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return np.matmul(starting_matrix, A)

# ☆ 
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    # Initiialize initial positions for p_e and p_m
    pe_0 = np.array([x_e, 0])
    pm_0 = np.array([x_m, 0])   

    # Initialize time space
    t = np.linspace(0, T, 200)
    
    # Define a helper rotated function
    def rotated(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    
    # Initialize Earth, and Moon positions at a given time t
    p_e_t = []
    p_m_t = []
    
    for value in t:
        # Calculate p_e(t) at each value
        earth_pos = np.dot(rotated(omega_e * value), pe_0)  # Rotate p_e(0) counterclockwise about the origin by t*omega_e radians.
        p_e_t.append(earth_pos)
    
        # Calculate the position of the moon relative to the earth at a given value of time t
        moon_rel_pos = np.dot(rotated(omega_m * value), pm_0 - pe_0)

         # Calculate p_m(t) at each value
        moon_abs_pos = earth_pos + moon_rel_pos
        p_m_t.append(moon_abs_pos)
    
    # Convert lists to array for plotting purposes
    p_e_t = np.array(p_e_t)
    p_m_t = np.array(p_m_t)
    
    # Plot our functions p_e(t) and p_m(t) over the time interval defined above
    plt.plot(p_e_t[:, 0], p_e_t[:, 1], label="Earth", color="blue")
    plt.plot(p_m_t[:, 0], p_m_t[:, 1], label="Moon", color="orange")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title("Trajectories of the Earth and Moon")
    
    plt.show()


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p)]
                                    for i in range(m)]

def method_timings():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    # Initialize n values, and times to be added
    n_values = list(range(10, 300, 10))
    matrix_vector_times = []
    matrix_matrix_times = []

    # Gather our times to compute the matrix each way
    for n in n_values:
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)
    
        m_v_start = time.time()
        matrix_vector_product(A,x)
        m_v_end = time.time()
        matrix_vector_times.append(m_v_end - m_v_start)

        m_m_start = time.time()
        matrix_matrix_product(A,B)
        m_m_end =  time.time()
        matrix_matrix_times.append(m_m_end - m_m_start)
    
    # Plot our graphs
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.plot(n_values, matrix_vector_times, color = 'blue')
    ax1.set_title("Matrix-Vector Multiplication")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Seconds")
    ax1.set_ylim(0, 0.003)
    
    ax2.plot(n_values, matrix_matrix_times, color = 'orange')
    ax2.set_title("Matrix-Matrix Multiplication")
    ax2.set_xlabel('n')
    ax2.set_ylim(0, 1.0)
    
    plt.show()

    
def times_and_plots():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    # Initialize n values, and times to be added
    n_values = list(range(10, 300, 10))
    matrix_vector_times = []
    matrix_matrix_times = []
    m_v_dot = []
    m_m_dot = []

    # Gather our times to compute the matrix each way
    for n in n_values:
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)
    
        m_v_start = time.time()
        matrix_vector_product(A,x)
        m_v_end = time.time()
        matrix_vector_times.append(m_v_end - m_v_start)

        m_m_start = time.time()
        matrix_matrix_product(A,B)
        m_m_end =  time.time()
        matrix_matrix_times.append(m_m_end - m_m_start)

        m_v_dot_start = time.time()
        np.dot(A,x)
        m_v_dot_end = time.time()
        m_v_dot.append(m_v_dot_end - m_v_dot_start)
        
        m_m_dot_start = time.time()
        np.dot(A,B)
        m_m_dot_end = time.time()
        m_m_dot.append(m_m_dot_end - m_m_dot_start)
    
    # Plot our graphs
    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.plot(n_values, matrix_vector_times, label = "matrixvector", color = 'blue')
    ax1.plot(n_values, matrix_matrix_times, label = "matrix-matrix", color = 'orange')
    ax1.plot(n_values, m_v_dot, label = "matrix-vector_np.dot()")
    ax1.plot(n_values, m_m_dot, label = "matrix-matrix_np.dot()", color = 'green')
    ax1.set_title('Linear Scale')
    ax1.set_xlabel("n")
    ax1.set_ylabel("Seconds")
    ax1.legend(markerscale = 0.5)

    
    ax2.loglog(n_values, matrix_vector_times, base = 2, label = "matrixvector", color = 'blue')
    ax2.loglog(n_values, matrix_matrix_times, base = 2, label = "matrix-matrix", color = 'orange')
    ax2.loglog(n_values, m_v_dot, base = 2, label = "matrix-vector_np.dot()")
    ax2.loglog(n_values, m_m_dot, base = 2, label = "matrix-matrix_np.dot()", color = 'green')
    ax2.set_title("Log-Log Scale")
    ax2.set_xlabel("n")
    ax2.legend(loc="upper left")
    
    plt.show()
    
    