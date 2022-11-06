import numpy as np


def gram_schmidt(A : np.ndarray):
    
    M = A.copy()
    (n, m) = M.shape
    
    for i in range(m):
        
        q = M[:, i] # i-th column of A
        
        for j in range(i):
            q = q - np.dot(M[:, j], M[:, i]) * M[:, j]
        
        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")
        
        # normalize q
        q = q / np.sqrt(np.dot(q, q))
        
        # write the vector back in the matrix
        M[:, i] = q

    return M
