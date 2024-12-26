
import svd_image_compression as sv
import numpy as np
import numpy.linalg as nla
import pytest
from scipy import linalg as la

def test_compact_svd():
        """Unit test for the algorithm to compute the compact SVD of a matrix"""
        m = 7 
        n = 6
        A = np.random.randint(1, 10, (m, n)).astype(float)
        U, sigma, V =  sv.compact_svd(A) 

        assert np.allclose(U@np.diag(sigma)@V, A) is True, "Incorrect truncated SVD"
        assert np.allclose(U.T @ U, np.identity(n)) is True, "U is not orthonormal"
        assert np.allclose(V.T @ V, np.identity(n)) is True, "V is not orthonormal"
        assert nla.matrix_rank(A) == len(sigma), "Number of nonzero singular values is not equal to rank of A"
        
def test_svd_approx():
    """Unit test for approximating the rank S SVD approximation of a matrix A"""
    # Set up matrix
    A = np.random.random((20, 20))
    s = 5
    A_s, num_entries = sv.svd_approx(A, s)

    # Test that Rank of A_s = s
    assert nla.matrix_rank(A_s) == s, 'Rank of A_s must = s'

    # Test that A_s has same shape of s
    assert A_s.shape == A.shape, "A_s and s must have same shape"

    # Make sure we have correct number of entries
    U, Sigma, Vh = la.svd(A)
    correct_entries = U[:, :s].size + Sigma[:s].size + Vh[:s, :].size
    assert num_entries == correct_entries, "Number of entries is not correct"

    # Check that the approximation is reasonable
    possible_error = np.linalg.norm(A - A_s, ord='fro')
    original_norm = np.linalg.norm(A, ord='fro')
    assert .5 * original_norm - possible_error < 4
    
    # Test that ValueError is raised when s is greater than rank of matrix
    with pytest.raises(ValueError):
        B = np.array([[1, 1], [1, 1]])
        sv.svd_approx(B, s)
                