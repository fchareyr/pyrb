import numpy as np
from typing import Any
import quadprog


def to_column_matrix(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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


def to_array(x: np.ndarray[Any, Any] | None) -> np.ndarray[Any, Any] | None:
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


def quadprog_solve_qp(
    p: np.ndarray[Any, Any],
    q: np.ndarray[Any, Any],
    g: np.ndarray[Any, Any] | None = None,
    h: np.ndarray[Any, Any] | None = None,
    a: np.ndarray[Any, Any] | None = None,
    b: np.ndarray[Any, Any] | None = None,
    bounds: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Quadprog helper for solving quadratic programming problems.

    Args:
        p: Quadratic term matrix.
        q: Linear term vector.
        g: Inequality constraint matrix (optional).
        h: Inequality constraint vector (optional).
        a: Equality constraint matrix (optional).
        b: Equality constraint vector (optional).
        bounds: Variable bounds (optional).

    Returns:
        Solution vector from quadratic programming solver.
    """
    n = p.shape[0]
    if bounds is not None:
        identity = np.eye(n)
        lower_bound = -identity
        upper_bound = identity
        if g is None:
            g = np.vstack([lower_bound, upper_bound])
            bounds_0 = to_array(bounds[:, 0])
            bounds_1 = to_array(bounds[:, 1])
            if bounds_0 is not None and bounds_1 is not None:
                h = np.array(np.hstack([-bounds_0, bounds_1]))
        else:
            g = np.vstack([g, lower_bound, upper_bound])
            bounds_0 = to_array(bounds[:, 0])
            bounds_1 = to_array(bounds[:, 1])
            if h is not None and bounds_0 is not None and bounds_1 is not None:
                h = np.array(np.hstack([h, -bounds_0, bounds_1]))

    qp_a = q  # because  1/2 x^T G x - a^T x
    qp_g = p
    if a is not None:
        assert g is not None and h is not None and b is not None
        qp_c = -np.vstack([a, g]).T
        qp_b = -np.hstack([b, h])
        meq = a.shape[0]
    else:  # no equality constraints
        qp_c = -g.T if g is not None else np.array([[]])
        qp_b = -h if h is not None else np.array([])
        meq = 0
    return np.array(quadprog.solve_qp(qp_g, qp_a, qp_c, qp_b, meq)[0])


def proximal_polyhedra(
    y: np.ndarray[Any, Any],
    c: np.ndarray[Any, Any],
    d: np.ndarray[Any, Any],
    bound: np.ndarray[Any, Any],
    a: np.ndarray[Any, Any] | None = None,
    b: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Wrapper for projecting a vector on the constrained set.

    Args:
        y: Vector to project.
        c: Constraint matrix.
        d: Constraint vector.
        bound: Variable bounds.
        a: Additional constraint matrix (optional).
        b: Additional constraint vector (optional).

    Returns:
        Projected vector on the constrained set.
    """
    n = len(y)
    return quadprog_solve_qp(
        np.eye(n), np.array(y), np.array(c), np.array(d), a=a, b=b, bounds=bound
    )
