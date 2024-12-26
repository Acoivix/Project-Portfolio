"""Unit testing file for Least Squares and Computing Eigenvalues """


import lstsq_eigs
import numpy as np

def test_qr_algorithm():
    """
    Write at least one unit test for problem 6, the qr algorithm function.
    """
    # Test 1 by 1
    A1 = np.array([[1]])
    correct_eigenvalue = 1
    my_eigenvalue = lstsq_eigs.qr_algorithm(A1)
    assert np.allclose(correct_eigenvalue, my_eigenvalue) is True, "Failed on 1 by 1 matrix"

    # Test 2 by 2
    A2 = np.array([[2, 1],[1, 2]])
    expected_eigs = np.array([3, 1])
    my_evals = lstsq_eigs.qr_algorithm(A2)
    assert np.allclose(expected_eigs, my_evals) is True, "Failed on 2 by 2 matrix"


def test_power_method():
    #Sets up test cases
    A = np.array([[1, 1], [1, 1]])
    B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    C = np.array([[2, 2], [1, 3]])
    
    Aval, Avec = lstsq_eigs.power_method(A)
    Bval, Bvec = lstsq_eigs.power_method(B)
    Cval, Cvec = lstsq_eigs.power_method(C)
    
    #Checks if it finds the appropriate eigenvalue
    assert abs(Aval - 2) < 1e-5, "Incorrect eigenvalue"
    assert abs(Bval - 3) < 1e-5, "Incorrect eigenvalue"
    assert abs(Cval - 4) < 1e-5, "Incorrect eigenvalue"
    
    #Checks if it finds an eigenvector that works
    assert np.linalg.norm(A @ Avec - Aval * Avec) < 1e-3, "Incorrect vector"
    assert np.linalg.norm(B @ Bvec - Bval * Bvec) < 1e-3, "Incorrect vector"
    assert np.linalg.norm(C @ Cvec - Cval * Cvec) < 1e-3, "Incorrect vector"