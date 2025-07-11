import numpy as np
import quadprog


def to_column_matrix(x):
    """Return x as a matrix columns.

    Args:
        x: Input array to convert to column matrix format.

    Returns:
        Array reshaped as a column matrix.

    Raises:
        ValueError: If x is not a vector.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim == 2:
        if x.shape[1] != 1:
            x = x.T
        if x.shape[1] == 1:
            return x
        else:
            raise ValueError("x is not a vector")
    else:
        raise ValueError("x is not a vector")
    return x


def to_array(x):
    """Turn a columns or row matrix to an array.

    Args:
        x: Input matrix to convert to array format.

    Returns:
        Array squeezed from matrix format, or None if x is None.
    """
    if x is None:
        return None
    elif (len(x.shape)) == 1:
        return x

    if x.shape[1] != 1:
        x = x.T
    return np.squeeze(np.asarray(x))


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None, bounds=None):
    """Quadprog helper for solving quadratic programming problems.

    Args:
        P: Quadratic term matrix.
        q: Linear term vector.
        G: Inequality constraint matrix (optional).
        h: Inequality constraint vector (optional).
        A: Equality constraint matrix (optional).
        b: Equality constraint vector (optional).
        bounds: Variable bounds (optional).

    Returns:
        Solution vector from quadratic programming solver.
    """
    n = P.shape[0]
    if bounds is not None:
        identity = np.eye(n)
        LB = -identity
        UB = identity
        if G is None:
            G = np.vstack([LB, UB])
            h = np.array(np.hstack([-to_array(bounds[:, 0]), to_array(bounds[:, 1])]))
        else:
            G = np.vstack([G, LB, UB])
            h = np.array(
                np.hstack([h, -to_array(bounds[:, 0]), to_array(bounds[:, 1])])
            )

    qp_a = q  # because  1/2 x^T G x - a^T x
    qp_G = P
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraints
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def proximal_polyhedra(y, C, d, bound, A=None, b=None):
    """Wrapper for projecting a vector on the constrained set.

    Args:
        y: Vector to project.
        C: Constraint matrix.
        d: Constraint vector.
        bound: Variable bounds.
        A: Additional constraint matrix (optional).
        b: Additional constraint vector (optional).

    Returns:
        Projected vector on the constrained set.
    """
    n = len(y)
    return quadprog_solve_qp(
        np.eye(n), np.array(y), np.array(C), np.array(d), A=A, b=b, bounds=bound
    )
