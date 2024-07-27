import sys
import numpy as np

from pm4py.util.lp import solver as lp_solver
from cvxopt import matrix

def derive_heuristic(incidence_matrix, cost_vec, x, t, h):
    x_prime = x.copy()
    x_prime[incidence_matrix.transitions[t]] -= 1
    return max(0, h - cost_vec[incidence_matrix.transitions[t]]), x_prime


def compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix, marking,
                                        fin_vec, variant):
    m_vec = incidence_matrix.encode_marking(marking)
    b_term = [i - j for i, j in zip(fin_vec, m_vec)]
    b_term = np.matrix([x * 1.0 for x in b_term]).transpose()
    b_term = matrix(b_term)
    parameters_solving = {"solver": "glpk"}
    sol = lp_solver.apply(cost_vec,
                          g_matrix,
                          h_cvx,
                          a_matrix,
                          b_term,
                          parameters=parameters_solving,
                          variant=variant)
    prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
    points = lp_solver.get_points_from_sol(sol, variant=variant)
    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = points if points is not None else [0.0] * len(sync_net.transitions)
    return prim_obj, points


def trust_solution(x):
    for v in x:
        if v < -0.001:
            return False
    return True
