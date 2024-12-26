import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import cmath


def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.
    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    # Obtain A, R
    Q, R = la.qr(A, mode='economic')

    # Obtain Q^T and 
    Qt_b = np.dot(Q.T, b)

    # Solve Rx = Q^tb
    x = la.solve_triangular(R, Qt_b)

    return x


def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    # Load and format data
    data = np.load('housing.npy')
    years = data[:, 0] 
    prices = data[:, 1]

    # Construct A and b
    A = np.column_stack((years,  np.ones_like(years)))
    b = prices

    # Find least squares solution
    x = least_squares(A, b)

    # Plot data points as scatter plot
    plt.scatter(years, prices, label = "Individual data points")

    # Construct least squares line
    m, c = x
    y = m * years + c                # y = mx + b

    # Plot least squares line
    plt.plot(years, y, label = "line_of_best_fit")

    plt.title("Changes in Housing pricing")
    plt.xlabel("Years passed")
    plt.ylabel("Housing price index")
    plt.legend()

    plt.show()


def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    # Load and format data
    data = np.load("housing.npy")
    years = data[:, 0]
    prices = data[:, 1]

    lower_bd = years.min()
    upper_bd = years.max()

    # Create domain based on bounds
    domain = np.linspace(lower_bd, upper_bd, 300)
    
    # Setup plots    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    degrees = [3, 6, 9, 12]

    for i, degree in enumerate(degrees):
        # Get current subplot
        row, col = i // 2, i % 2
        current_ax = axes[row, col]

        # Construct vander matrix
        A = np.vander(years, degree + 1)

        # Solve for least squares solution
        coeffs = la.lstsq(A, prices)[0]

        # Get refined polynomial on domain
        refined_A = np.vander(domain, degree + 1)
        fit_values = refined_A @ coeffs

        # Plot original data
        current_ax.scatter(years, prices, label = "Individual data points")
        
        # Plot polynomial fit
        current_ax.plot(domain, fit_values, label = "Line of best fit")
        current_ax.set_title(f'Polynomial Fit (Degree {degree})')
        current_ax.set_xlabel('Year')
        current_ax.set_ylabel('Housing Prices')
        current_ax.legend()

    fig.suptitle('Polynomial fits of different degrees for change in Housing Prices', fontsize=16)
    plt.legend()
    plt.show()

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t,  label = "Elipse of best fit")
    plt.gca().set_aspect("equal", "datalim")

def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    # Load data
    xk, yk = np.load("ellipse.npy").T

    # Construct A and b    
    A = np.column_stack((xk**2, xk, xk * yk, yk, yk**2))
    b_vector = np.ones_like(xk)
    print(b_vector)

    # Solve the least squares problem to find ellipse parameters
    a, b, c, d, e = la.lstsq(A, b_vector)[0]

    # Plot scatter plot of original data
    plt.scatter(xk, yk, label = "Individual data points")
    # Plot ellipse
    plot_ellipse(a, b, c, d, e)
    
    plt.legend()
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Ellipse of best fit")
    plt.show()

def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    _, n  = np.shape(A)        # A is a square so m = n
    x0 = np.random.rand(n)      # Random vector of length n  
    x0 = x0 / la.norm(x0)       # Normalize x0

    x_k = x0    

    for k in range(N):
        x_k_1 = A @ x_k                       # x_k_1 denotes x_k+1
        x_k_1 = x_k_1 / np.linalg.norm(x_k_1) 

        # Check convergence
        if np.linalg.norm(x_k_1 - x_k) < tol:
            x_k = x_k_1
            break
        
        # Compute the eigenvalue
        eig = np.dot(x_k_1, A @ x_k_1)
        x_k = x_k_1

    return eig, x_k_1  


def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    _, n  = np.shape(A)
    # Put A in Upper Hessenberg form
    S = la.hessenberg(A)
    
    for k in range(N):
        # Get the QR Decomposition of Ak
        Q, R = la.qr(S, mode = 'economic')
        # Recombine Rk and QK into Ak+1
        S = R @ Q

     # Initialize empty list of eigenvalues
    eigs = []
    i = 0

    while i < n:
        # If Si is 1 by 1
        if i == n - 1 or abs(S[i+1, i]) < tol:
            eigs.append(S[i, i])
            i += 1
        # If Si is 2 by 2
        else:
            # Construct matrix
            a, b = S[i][i], S[i][i+1]
            c, d = S[i+1][i], S[i+1][i+1]

            # Calculate eigenvalues of matrix
            det = a*d - b*c
            trace = a + d
            discriminant =  cmath.sqrt(trace**2 - 4 * det)
            lambda_1 = (trace + discriminant) / 2
            lambda_2 = (trace - discriminant) / 2

            eigs.append(lambda_1)
            eigs.append(lambda_2)

            # Move to next Si
            i += 2
       
    return np.array(eigs)