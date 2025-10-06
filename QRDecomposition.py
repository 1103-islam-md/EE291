import numpy as np

#USER INPUT SECTION
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))
print(f"Enter {rows * cols} matrix elements separated by spaces:")
matrixnum = list(map(float, input().split()))
if len(matrixnum) != rows * cols:
    raise ValueError(f"Expected {rows*cols} numbers, got {len(matrixnum)}")

A = np.array(matrixnum, dtype=float).reshape(rows, cols)
print("\nMatrix A:")
print(A)
"""
This section lets the user choose how big their matrix is by entering
the number of rows and columns. The program then reshapes the user's
input values into an actual NumPy matrix. This makes the code flexible,
so it can perform QR decomposition on any size of matrix the user inputs.
"""
#QR DECOMPOSITION

""" The QR decomposition breaks matrix A into two parts:
    - Q: an orthogonal matrix (its columns are perpendicular)
    - R: an upper triangular matrix
    This is done using a process called Householder reflections.
"""
def qr_decomposition(A, reduced=False, verbose=True):
    A = np.array(A, dtype=float, copy=True)
    m, n = A.shape
    k = min(m, n)
    R = A.copy()
    Q = np.eye(m)

    for p in range(k):
        x = R[p:, p]
        normx = np.linalg.norm(x)
        if normx == 0.0:
            if verbose:
                print(f"\nIteration {p+1} (skipped: column already zero below diag)")
            continue

        e1 = np.zeros_like(x); e1[0] = 1.0
        v = x + np.copysign(normx, x[0]) * e1
        vnorm = np.linalg.norm(v)
        if vnorm == 0.0:
            if verbose:
                print(f"\nIteration {p+1} (skipped: v ~ 0)")
            continue
        #u = v/||v||
        u = v / vnorm
        #H = I - 2uuT
        H_sub = np.eye(len(u)) - 2.0 * np.outer(u, u)
        H_p = np.eye(m)
        H_p[p:, p:] = H_sub

        R = H_p @ R
        Q = Q @ H_p
        """
        Each iteration creates a reflection (H_p) that zeroes out
        the values below the diagonal in one column of R. The Q
        matrix collects all of these reflections so that, by the
        end, A = Q * R.
        """
        if verbose:
            print(f"\n===Iteration {p+1} ===")
            print(f"u{p+1} = {u}")
            print(f"H{p+1} (Householder Matrix) =\n{H_p}")
            print(f"Q{p+1} =\n{Q}")
            print(f"R{p+1} =\n{R}")
            print(f"New A (for next iteration) =\n{R}")
            print("==========")

    if reduced:
        return Q[:, :k], R[:k, :]
    return Q, R

#QR ALGORITHM DRIVER
"""
The QR algorithm uses repeated QR decompositions to find eigenvalues.
It repeatedly computes A = RQ until A becomes almost upper triangular.
The diagonal entries of this final matrix are the eigenvalues.
"""
def is_upper_triangular(A, tol=1e-10):
    return np.allclose(A, np.triu(A), atol=tol)

def deflate_if_possible(A, tol=1e-10):
    """Removes small sub-diagonal values to simplify the matrix (deflation)."""
    n = A.shape[0]
    eigvals = []
    while n > 1 and abs(A[n-1, n-2]) < tol:
        eigvals.append(A[n-1, n-1])
        A = A[:n-1, :n-1]
        n -= 1
    return A, eigvals

def wilkinson_shift_1x1(A):
    """Uses the bottom-right entry as a shift value to improve accuracy."""
    return A[-1, -1]

def eigenvalues_from_2x2(B):
    """Finds eigenvalues of a 2x2 block, which may be real or complex."""
    a, b = B[0,0], B[0,1]
    c, d = B[1,0], B[1,1]
    tr = a + d
    det = a*d - b*c
    disc = tr*tr - 4*det
    if disc >= 0:
        r = np.sqrt(disc)
        return [(tr + r)/2, (tr - r)/2]
    else:
        r = np.sqrt(-disc)
        return [complex(tr/2,  r/2), complex(tr/2, -r/2)]

def QRAlgorithm(A, tol=1e-10, max_iter=5000, use_shift=True, verbose=False):
    """Applies the QR Algorithm iteratively to compute eigenvalues."""
    A = np.array(A, dtype=float, copy=True)
    eigvals = []
    k = 0
    while A.shape[0] > 2 and k < max_iter:
        A, newly = deflate_if_possible(A, tol)
        eigvals.extend(newly)
        if A.shape[0] <= 2:
            break

        mu = wilkinson_shift_1x1(A) if use_shift else 0.0
        A_shifted = A - mu * np.eye(A.shape[0])

        Q, R = qr_decomposition(A_shifted, reduced=True, verbose=False)
        A = R @ Q + mu * np.eye(A.shape[0])

        k += 1
        if verbose and k % 25 == 0:
            print(f"Iteration {k}: off-diagonal = {np.linalg.norm(np.tril(A, -1)):.2e}")

        if np.linalg.norm(np.tril(A, -1), ord='fro') < tol:
            break

    n = A.shape[0]
    if n == 1:
        eigvals.append(A[0,0])
    elif n == 2:
        eigvals.extend(eigenvalues_from_2x2(A))
    else:
        eigvals.extend(np.diag(A))
    return eigvals

    """the program prints out the orthogonal matrix Q, the upper
triangular matrix R, and the eigenvalues of the input matrix. This combines
the QR decomposition and the QR algorithm into one complete system that
can handle any square or rectangular matrix the user provides.
"""

#RUN DRIVER TEST
print("\n=== QR Algorithm (Eigenvalues) ===")
eigs = QRAlgorithm(A, tol=1e-12, verbose=False)
print("Eigenvalues of input matrix:")
for val in eigs:
    print(val)
print("============")

#Final matrices for Part 1 reporting
Q, R = qr_decomposition(A, reduced=False, verbose=False)
print("\nFinal Matrix Q:")
print(Q)
print("\nFinal Matrix R:")
R_print = np.triu(R)
R_print[np.isclose(R_print, 0, atol=1e-12)] = 0.0
print(R_print)
