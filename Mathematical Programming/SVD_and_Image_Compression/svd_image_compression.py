import numpy as np
from imageio.v3 import imread
from scipy import linalg as la
from matplotlib import pyplot as plt


def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    # Calculate the eigenvalues and eigenvectors of A^HA
    AHA = A.conj().T @ A
    lamda, V = la.eig(AHA)

    # Calculate the singular values of A
    sigma = np.sqrt(np.abs(lamda))
    
    # Sort the singular values from greatest to least
    sorted_indices = np.argsort(sigma)[::-1] 
    sigma = sigma[sorted_indices]

    # Sort the eigenvectors the same way
    V = V[:, sorted_indices]

    # Count the number of nonzero singular values
    r = np.count_nonzero(sigma > tol)

    # Keep only the positive singular values
    sigma_1 = sigma[:r]

    # Keep only the corresponding eigenvectors
    V1 = V[:, :r]

    # Construct U with array broadcasting
    U1 = (A @ V1) / sigma_1

    return U1, sigma_1, V1.conj().T


def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    # Generate matrix x
    theta = np.linspace(0, 2*np.pi, 200)
    S = np.vstack([np.cos(theta), np.sin(theta)])
    
    # Define standard basis vectors
    E = np.array([[1, 0], [0, 1]])
    
    # Compute the SVD of A
    U, Sigma, Vh = la.svd(A)
    Sigma = np.diag(Sigma)

    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('SVD Transformation With Intermediate Steps')
    
    # Setup helper plot vector function
    def plot_vectors(ax, S, E, title):
        ax.plot(S[0], S[1])  
        for vector in E.T:
            ax.arrow(0, 0, vector[0], vector[1], head_width=0.05, color='orange', length_includes_head=True)
        ax.axis('equal')
        ax.set_title(title)

    # Original
    plot_vectors(axes[0, 0], S, E, "S")

    # After applying V^H
    VhS = Vh @ S
    VhE = Vh @ E
    plot_vectors(axes[0, 1], VhS, VhE, "V^HS")

    # After applying sigma
    SVhS = Sigma @ VhS
    SVhE = Sigma @ VhE
    plot_vectors(axes[1, 0], SVhS, SVhE, "sigmaV^HS")

    # After applying U
    USVhS = U @ SVhS
    USVhE = U @ SVhE
    plot_vectors(axes[1, 1], USVhS, USVhE, "UsigmaV^HS")

    plt.tight_layout()
    plt.show()


def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    # Compute the compact SVD of A
    U, Sigma, Vh = la.svd(A)

    # Raise ValueError if s is greater than the number of nonzero singular values of A
    A_rank = np.linalg.matrix_rank(A)
    if s > A_rank:
        raise ValueError(f"{s} cannot be greater than rank of {A}")
    
    # Form the truncated SVD, keeping the first s components of each component
    U_s = U[:, :s]
    Sigma_s = Sigma[:s]
    Vh_s = Vh[:s, :]

    # Get the best rank s approximation of As
    A_s = U_s @ np.diag(Sigma_s) @ Vh_s

    # Get the number of entries needed to store the truncated form
    entries_num = U_s.size + Sigma_s.size + Vh_s.size

    return A_s, entries_num 


def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    # Compute the compact SVD of A
    U, Sigma, Vh = la.svd(A)

    # Raise ValueError if epsilon <= to the smallest singular value of A
    smallest_sing_value = Sigma[-1]
    if err <= smallest_sing_value:
        raise ValueError("Error cannot be less than or equal to smallest singular value")
    
    # Find the s such that sigma_s+1 is the largest sigular value less than epsilon
    s = np.argmax(Sigma < err)

    # Form the truncated SVD, keeping the first s components of each component
    U_s = U[:, :s]
    Sigma_s = Sigma[:s]
    Vh_s = Vh[:s, :]

    # Get the best rank s approximation of As
    A_s = U_s @ np.diag(Sigma_s) @ Vh_s

    # Get the number of entries needed to store the truncated form
    entries_num = U_s.size + Sigma_s.size + Vh_s.size

    return A_s, entries_num 
    

def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    # Read image in and check if its colored or not
    image = imread(filename)
    color = 0
    if image.ndim == 2:     # If image is grayscale
        pass
    else:
        color += 1          # Image is colored
    
    # Obtain reank approximations
    if color == 0:          # If image is grayscale
        combined_matrix, new_total_entries = svd_approx(image, s)
    else:
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
        
        # Obtain rank approximations and num of entries for each color channel
        R_s, R_storage = svd_approx(R, s)
        G_s, G_storage = svd_approx(G, s)
        B_s, B_storage = svd_approx(B, s)

        # Put them back together in new three dimensional array
        combined_matrix = np.dstack((R_s, G_s, B_s))

        # Record new total entries
        new_total_entries = R_storage + G_storage + B_storage

    # Get difference in entries between original image and new image
    og_entries = image.size
    difference_in_entries = og_entries - new_total_entries

    # Setup plot
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f"Difference in entries: {difference_in_entries}")

    # Plot original
    scaled_image = image / 255
    plt.subplot(1, 2, 1)
    plt.imshow(scaled_image, cmap="gray" if color == 0 else None)
    plt.axis('off')
    plt.title("OG image")

    # Plot approximated image
    scaled_adjusted_image = combined_matrix / 255
    plt.subplot(1, 2, 2)
    plt.imshow(scaled_adjusted_image, cmap="gray" if color == 0 else None)
    plt.axis('off')
    plt.title(f"Image with rank approximation {s}")

    plt.tight_layout()
    plt.show()


