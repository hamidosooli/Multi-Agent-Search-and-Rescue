"""
This module includes different methods for calculating
the optimal weights for the Perron matrix in a distributed
consensus.

x(t+1) = P x(t)

The modules use A the incidence matrix of the network.

The following function have been implemented:

1- max_degree_weights - uses maximum degree [constant]
2- metropolis_hastings_weights - uses local max weights
3- fastest_averaging_constant_weight - uses [constant]
4- fdla_weights - fastest distributed linear averaging edge_weights
5- fdla_weights_symmetric - uses fdla when P is symmetric

Adopted from code in: https://github.com/cvxr/CVX

Code: Reza Ahmadzadeh 2022
"""
from lib2to3.pytree import convert

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import laplacian


def max_degree_weights(A):
    """
    returns maximum degree weights 
    Maximum-degree edge weights are all equal to one over the maximum
    degree of the nodes in the graph.

    Input:
        A graph described by the incidence matrix A size (n,m)
        where n is the number of nodes, and m is the number of edges. 
        Each column of A has exactly one +1 % and one -1.

    Outputs:
        w_md: max degree weight vector size (m)
        W_md: max degree weight matrix size (n,n)
        rho: spectral radius
    """
    
    n, m = np.shape(A)
    I = np.eye(n)
    Lunw = A @ A.T                  # unweighted Laplacian
    degs = np.diag(Lunw)
    max_deg = np.max(degs)
    alpha_md = 1 / max_deg
    w_md = alpha_md * np.ones(m)    # weight vector
    W_md = I - alpha_md * Lunw      # compute the Weight matrix (P)
    # compute the norm
    rho = np.linalg.norm(I - A@np.diag(w_md)@A.T - (1/n)*np.ones(n))
    # rho = np.linalg.norm(W_md - (1/n)*np.ones(n)) ## more optimized?? 
    
    return alpha_md, w_md, W_md, rho


def fastest_averaging_constant_weight(A):
    """
    calculates a vector of the best constant edge weights
    Input:
        A graph described by the incidence matrix A size (n,m)
        where n is the number of nodes, and m is the number of edges. 
        Each column of A has exactly one +1 % and one -1.

    Outputs:
        w_fa: fastest averaging weight vector size (m)
        W_fa: fastest averaging weight matrix size (n,n)
        rho: spectral radius
    
        The best constant edge weight is the inverse of the average of
        the second smallest and largest eigenvalues of the unweighted
        Laplacian:
        W = 2/( lambda_2(A*A') + lambda_n(A*A') )
        RHO is computed from the weights W as follows:
        RHO = max(abs(eig( eye(n,n) - (1/n)*ones(n,n) - A*W*A' ))).
    """
    
    n, m = np.shape(A)
    I = np.eye(n)
    Lunw = A @ A.T                      # unweighted Laplacian
    eigvals, _ = np.linalg.eig(Lunw)    # eigenvalues
    eigvals_sorted = np.sort(eigvals)   # sort in ascending order
    alpha_fa = 2 / (eigvals_sorted[1] + eigvals_sorted[-1])
    w_fa = alpha_fa * np.ones(m)        # build the weight vector
    W_fa = I - alpha_fa * Lunw      # compute the Weight matrix (P)
    # compute the norm
    rho = np.linalg.norm(I - A@np.diag(w_fa)@A.T - (1/n)*np.ones(n))
    # rho = np.linalg.norm(W_fa - (1/n)*np.ones(n)) ## more optimized ??

    return alpha_fa, w_fa, W_fa, rho



def metropolis_hastings_weights(A):
    """
    calculates a vector of the Metropolis-Hastings edge weights.
    The M.-H. weight on an edge is one over the maximum of the
    degrees of the adjacent nodes.

    Input:
        A graph described by the incidence matrix A size (n,m)
        where n is the number of nodes, and m is the number of edges. 
        Each column of A has exactly one +1 % and one -1.

    Outputs:
        w_mh: fastest averaging weight vector size (m)
        W_mh: fastest averaging weight matrix size (n,n)
        rho: spectral radius

        RHO is computed from the weights W as follows:
        RHO = max(abs(eig( eye(n,n) - (1/n)*ones(n,n) - A*W*A' ))).
    """
    
    n, m = np.shape(A)
    I = np.eye(n)
    Lunw = A @ A.T                          # unweighted Laplacian
    degs = np.diag(Lunw)
    mh_degs = np.abs(A).T @ np.diag(degs)   
    w_mh = 1 / np.max(mh_degs, axis=1)      #weight vector
    W_mh = I - A @ np.diag(w_mh) @ A.T      # compute the Weight matrix (P)
    # compute the norm    
    rho = np.linalg.norm(I - A @ np.diag(w_mh) @ A.T - (1/n)*np.ones(n))
    # rho = np.linalg.norm(W_mh - (1/n)*np.ones(n)) ## more optimized??
    return w_mh, W_mh, rho



def fdla_weights_symmetric(A):
    """
    fastest_distributed_linear_averaging_edge_symmetric weights
    """
    n, m = np.shape(A)
    I = np.eye(n)
    J = I - (1/n)* np.ones((n,n))
    w = cp.Variable(shape=(m))
    L = cp.Variable((n,n), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.atoms.norm2(J - A @ cp.diag(w) @ A.T)))
    prob.solve()
    # print(f'status: {prob.status}')
    # if prob.status not in ["infeasible", "unbounded"]:
    #     # Otherwise, problem.value is inf or -inf, respectively.
    #     print("Optimal value: %s" % prob.value)
    #     for variable in prob.variables():
    #         print("Variable %s: value %s" % (variable.name(), variable.value))
    w_fdla = w.value
    W_fdla = I - A @ np.diag(w_fdla) @ A.T      # compute the Weight matrix (P)
    return w_fdla, W_fdla, prob.value



def fdla_weights(A):
    """
    calculates fastest_distributed_linear_averaging_edge_weights

    Input: a graph described by the incidence matrix A (n x m).
    where n is the number of nodes and m is the number of edges in the graph;
    each column of A has exactly one +1 and one -1.
    
     The FDLA edge weights are given by the SDP:
    
       minimize    s
       subject to  -s*I <= I - L - (1/n)11' <= s*I
    
     where the variables are edge weights w in R^m and s in R.
     Here L is the weighted Laplacian defined by L = A*diag(w)*A'.
     The optimal value is s, and is returned in the second output.
        
    """
    n, m = np.shape(A)
    I = np.eye(n)
    J = I - (1/n)* np.ones((n,n))
    w = cp.Variable(shape=(m))
    L = cp.Variable((n,n), symmetric=True)
    s = cp.Variable(1)
    constraints = [L == A @ cp.diag(w) @ A.T,
                   J - L <= s * I,
                   J - L >= -s * I
                   ]
    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve()
    print(f'status: {prob.status}')
    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % prob.value)
        for variable in prob.variables():
            print("Variable %s: value %s" % (variable.name(), variable.value))    
    return w.value, prob.value



def fmmc_weights(A):
    """
    Fastest Mixing Symmetric Markov Chain
    is a transition matrix, P, that minimizes
    the second largest eigenvalue magnitude of P.
    The smaller this value gets, the faster asymptotic
    convergence we have.

    In this case, P is symmetric, rows sum to 1, and
    it can be described as P = I - L
    """

    n, m = np.shape(A)
    I = np.eye(n)
    J = I - (1/n) * np.ones((n, n))
    s = cp.Variable()
    w = cp.Variable(m)
    L = cp.Variable((n, n), symmetric=True)
    constraints = [w >= 0,
                   cp.diag(L) <= 1,
                   L == A @ cp.diag(w) @ A.T,
                   J - L << s * I,
                   J - L >> -s * I]
    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve()
    # print(f'status: {prob.status}')
    # if prob.status not in ["infeasible", "unbounded"]:
    #     # Otherwise, problem.value is inf or -inf, respectively.
    #     print("Optimal value: %s" % prob.value)
    #     for variable in prob.variables():
    #         print("Variable %s: value %s" % (variable.name(), variable.value))    
    W_fmmc = I - A @ np.diag(w.value) @ A.T
    return w.value, W_fmmc, prob.value




def get_S(w, C):
    n, m = np.shape(C)
    S = np.zeros((n,n))
    for i in range(m):
        S += w[i] * np.outer(C[:,i], C[:,i].T)
    return S


def get_FG(w, C):
    n, m = np.shape(C)
    S = get_S(w, C)
    I = np.eye(n)
    J = (1/n) * np.ones((n,n))
    F = 2*I - S - J
    G = S + J
    return F, G


def get_grad(w, C):
    n, m = np.shape(C)
    F, G = get_FG(w, C)
    F_inv = np.linalg.inv(F)
    G_inv = np.linalg.inv(G)
    dfdw = np.zeros(m)
    for i in range(m):
        ij = np.nonzero(C[:,i])[0]
        dfdw[i] = 0.5 * (np.linalg.norm(F_inv[:,ij[0]] - F_inv[:,ij[1]]))**2 -\
            0.5 * (np.linalg.norm(G_inv[:,ij[0]] - G_inv[:,ij[1]]))**2 # eq from 3.1
    return dfdw


# eq 14
def cost(x, C):
    n, m = np.shape(C)
    W = np.eye(n) - get_S(x, C) # eq 7
    eigvals, _ = np.linalg.eig(W)    
    eigvals_sorted = np.sort(eigvals)[::-1]   # sort in ascending order (descending?)
    delta_ss = 0.0
    for i in range(1, n):
        delta_ss += 1 / (1 - eigvals_sorted[i]**2)
    return delta_ss


def obj_der(x, C):
    dfdw = get_grad(x, C)
    return dfdw


def lmsc_weights(C, obj=cost, jac=obj_der):
    # print('Warning: check the objective and the jacobian functions before running.')
    n, m = np.shape(C)
    x0, _, _ = metropolis_hastings_weights(C)   # initial guess
    res = minimize(obj, x0, args= (C,), method='BFGS' , \
                   jac=jac, options={'gtol': 1e-06})
    w_lmsc = res.x
    W_lmsc = np.eye(n) - C @ np.diag(w_lmsc) @ C.T
    return w_lmsc, W_lmsc


def generateP(A, kappa):
    dmax = np.max(np.sum(A, axis=0))
    L = laplacian(A, normed=False)
    M, _ = np.shape(A)
    I = np.eye(M)

    P = I - (kappa/dmax) * L
    return P


def main():
    # example 1
    E = np.array([
        [ 1,  1,  1,  0],
        [-1,  0,  0,  1],
        [ 0, -1,  0, -1],
        [ 0,  0, -1,  0],
    ])

    # house
    # E = np.array([[1, 0, 0, 0, -1, 0],
    #               [0, 0, 0, -1, 1, 1],
    #               [-1, 1, 0, 0, 0, -1],
    #               [0, 0, -1, 1, 0, 0],
    #               [0, -1, 1, 0, 0, 0]])

    # ring
    # E = np.array([
    #     [ 1,  0,  0,  0, -1],
    #     [-1,  1,  0,  0,  0],
    #     [ 0, -1,  1,  0,  0],
    #     [ 0,  0, -1,  1,  0],
    #     [ 0,  0,  0, -1,  1],
    # ])

    # all-to-all
    # E = np.array([
    #     [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0],
    #     [-1,  0,  0,  0,  1,  1,  1,  0,  0,  0],
    #     [ 0, -1,  0,  0, -1,  0,  0,  1,  1,  0],
    #     [ 0,  0, -1,  0,  0, -1,  0, -1,  0,  1],
    #     [ 0,  0,  0, -1,  0,  0, -1,  0, -1, -1],
    # ])

    # line
    # E = np.array([
    #     [ 1,  0,  0,  0],
    #     [-1,  1,  0,  0],
    #     [ 0, -1,  1,  0],
    #     [ 0,  0, -1,  1],
    #     [ 0,  0,  0, -1],
    # ])

    # star
    # E = np.array([
    #     [ 1,  1,  1,  1],
    #     [-1,  0,  0,  0],
    #     [ 0, -1,  0,  0],
    #     [ 0,  0, -1,  0],
    #     [ 0,  0,  0, -1],
    # ])

    # E = np.array([
    #     [ 1,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0],
    #     [-1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
    #     [ 0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #     [ 0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #     [ 0,  0,  0, -1,  1, -1,  0,  0,  0,  0,  0,  0,  0],
    #     [ 0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0, -1,  0],
    #     [ 0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  1],
    #     [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0],
    #     [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0],
    #     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1, -1],
    # ])

    # E = np.array([
    #     [  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #     [ -1,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #     [  0, -1,  0, -1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0],
    #     [  0,  0, -1,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0],
    #     [  0,  0,  0,  0, -1,  0,  0,  0,  0, -1, -1,  1,  1,  1,  0,  0,  0],
    #     [  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0,  0,  1,  1,  0],
    #     [  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1,  0,  1],
    #     [  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1, -1],
    # ])

    np.set_printoptions(linewidth=np.inf)

    w_, W, rho = max_degree_weights(E)
    print(f'max degree weights:\n{W}\n')

    w_, W, rho = metropolis_hastings_weights(E)
    print(f'metropolis-hastings weights:\n{W}\n')

    w_, W, rho_ = fastest_averaging_constant_weight(E)
    print(f'fastest averaging constant weight:\n{W}\n')
    
    w_, opt_val = fdla_weights(E)
    print(f'FDLA weights:\n{W}\n')
    
    w_, W, opt_val = fdla_weights_symmetric(E)
    print(f'FDLA symmetric weights:\n{W}\n')
    
    w_ = fmmc_weights(E)
    print(f'FMMC weights:\n{W}\n')

    w, W = lmsc_weights(E)
    print(f'LMSC Optimal weights:\n{W}\n')


if __name__=='__main__':
    main()
