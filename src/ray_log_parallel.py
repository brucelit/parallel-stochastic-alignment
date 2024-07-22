import ray
import heapq
import sys
import time
import numpy as np


from copy import copy
from enum import Enum
from cvxopt import matrix
from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri_net.utils import align_utils as utils, final_marking, initial_marking
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.lp import solver as lp_solver
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing
from tqdm import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer




class Parameters(Enum):
    PARAM_TRACE_COST_FUNCTION = 'trace_cost_function'
    PARAM_MODEL_COST_FUNCTION = 'model_cost_function'
    PARAM_SYNC_COST_FUNCTION = 'sync_cost_function'
    PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE = 'ret_tuple_as_trans_desc'
    PARAM_TRACE_NET_COSTS = "trace_net_costs"
    TRACE_NET_CONSTR_FUNCTION = "trace_net_constr_function"
    TRACE_NET_COST_AWARE_CONSTR_FUNCTION = "trace_net_cost_aware_constr_function"
    PARAM_MAX_ALIGN_TIME_TRACE = "max_align_time_trace"
    PARAM_MAX_ALIGN_TIME = "max_align_time"
    PARAMETER_VARIANT_DELIMITER = "variant_delimiter"
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    VARIANTS_IDX = "variants_idx"
    RETURN_SYNC_COST_FUNCTION = "return_sync_cost_function"


PARAM_TRACE_COST_FUNCTION = Parameters.PARAM_TRACE_COST_FUNCTION.value
PARAM_MODEL_COST_FUNCTION = Parameters.PARAM_MODEL_COST_FUNCTION.value
PARAM_SYNC_COST_FUNCTION = Parameters.PARAM_SYNC_COST_FUNCTION.value
PARAM_COST_TOLERANCE = 2
# Initialize Ray with the number of CPU cores


def get_best_worst_cost(petri_net, initial_marking, final_marking, parameters=None):
    """
    Gets the best worst cost of an alignment

    Parameters
    -----------
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking

    Returns
    -----------
    best_worst_cost
        Best worst cost of alignment
    """
    if parameters is None:
        parameters = {}
    trace = log_implementation.Trace()

    alignment, cost = apply(trace, petri_net, initial_marking, final_marking, parameters=parameters)

    return alignment, cost


def apply(trace: Trace, petri_net: PetriNet, initial_marking: Marking, final_marking: Marking,
          parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.AlignmentResult:
    """
    Performs the basic alignment search, given a trace and a net.

    Parameters
    ----------
    trace: :class:`list` input trace, assumed to be a list of events (i.e. the code will use the activity key
    to get the attributes)
    petri_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net to use in the alignment
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the Petri net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the Petri net
    parameters: :class:`dict` (optional) dictionary containing one of the following:
        Parameters.PARAM_TRACE_COST_FUNCTION: :class:`list` (parameter) mapping of each index of the trace to a positive cost value
        Parameters.PARAM_MODEL_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        model cost
        Parameters.PARAM_SYNC_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        synchronous costs
        Parameters.ACTIVITY_KEY: :class:`str` (parameter) key to use to identify the activity described by the events

    Returns
    -------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY)
    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    trace_net_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_CONSTR_FUNCTION, parameters,
                                                           None)
    trace_net_cost_aware_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_COST_AWARE_CONSTR_FUNCTION,
                                                                      parameters, construct_trace_net_cost_aware)

    if trace_cost_function is None:
        trace_cost_function = list(
            map(lambda e: utils.STD_MODEL_LOG_MOVE_COST, trace))
        parameters[Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

    if model_cost_function is None:
        # reset variables value
        model_cost_function = dict()
        sync_cost_function = dict()
        for t in petri_net.transitions:
            if t.label is not None:
                model_cost_function[t] = utils.STD_MODEL_LOG_MOVE_COST
                sync_cost_function[t] = utils.STD_SYNC_COST
            else:
                model_cost_function[t] = utils.STD_TAU_COST
        parameters[Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
        parameters[Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function

    if trace_net_constr_function is not None:
        # keep the possibility to pass TRACE_NET_CONSTR_FUNCTION in this old version
        trace_net, trace_im, trace_fm = trace_net_constr_function(trace, activity_key=activity_key)
    else:
        trace_net, trace_im, trace_fm, parameters[
            Parameters.PARAM_TRACE_NET_COSTS] = trace_net_cost_aware_constr_function(trace,
                                                                                     trace_cost_function,
                                                                                     activity_key=activity_key)

    alignment, optimal_cost = apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, parameters)

    return alignment,optimal_cost



def apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, parameters=None):
    """
        Performs the basic alignment search, given a trace net and a net.

        Parameters
        ----------
        trace: :class:`list` input trace, assumed to be a list of events (i.e. the code will use the activity key
        to get the attributes)
        petri_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net to use in the alignment
        initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the Petri net
        final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the Petri net
        parameters: :class:`dict` (optional) dictionary containing one of the following:
            Parameters.PARAM_TRACE_COST_FUNCTION: :class:`list` (parameter) mapping of each index of the trace to a positive cost value
            Parameters.PARAM_MODEL_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
            model cost
            Parameters.PARAM_SYNC_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
            synchronous costs
            Parameters.ACTIVITY_KEY: :class:`str` (parameter) key to use to identify the activity described by the events
            Parameters.PARAM_TRACE_NET_COSTS: :class:`dict` (parameter) mapping between transitions and costs

        Returns
        -------
        dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
        """
    if parameters is None:
        parameters = {}

    ret_tuple_as_trans_desc = exec_utils.get_param_value(Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE,
                                                         parameters, False)

    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    sync_cost_function = exec_utils.get_param_value(Parameters.PARAM_SYNC_COST_FUNCTION, parameters, None)
    trace_net_costs = exec_utils.get_param_value(Parameters.PARAM_TRACE_NET_COSTS, parameters, None)

    if trace_cost_function is None or model_cost_function is None or sync_cost_function is None:
        sync_prod, sync_initial_marking, sync_final_marking = construct(trace_net, trace_im,
                                                                        trace_fm, petri_net,
                                                                        initial_marking,
                                                                        final_marking,
                                                                        utils.SKIP)
        cost_function = utils.construct_standard_cost_function(sync_prod, utils.SKIP)
    else:
        revised_sync = dict()
        for t_trace in trace_net.transitions:
            for t_model in petri_net.transitions:
                if t_trace.label == t_model.label:
                    revised_sync[(t_trace, t_model)] = sync_cost_function[t_model]

        sync_prod, sync_initial_marking, sync_final_marking, cost_function = construct_cost_aware(
            trace_net, trace_im, trace_fm, petri_net, initial_marking, final_marking, utils.SKIP,
            trace_net_costs, model_cost_function, revised_sync)

    max_align_time_trace = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                      sys.maxsize)

    alignment, optimal_cost = apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function,
                                utils.SKIP, ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                max_align_time_trace=max_align_time_trace)

    return_sync_cost = exec_utils.get_param_value(Parameters.RETURN_SYNC_COST_FUNCTION, parameters, False)
    if return_sync_cost:
        # needed for the decomposed alignments (switching them from state_equation_less_memory)
        return alignment, cost_function

    return alignment,optimal_cost


def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, skip, ret_tuple_as_trans_desc=False,
                    max_align_time_trace=sys.maxsize):
    """
    Performs the basic alignment search on top of the synchronous product net, given a cost function and skip-symbol

    Parameters
    ----------
    sync_prod: :class:`pm4py.objects.petri.net.PetriNet` synchronous product net
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the synchronous product net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the synchronous product net
    cost_function: :class:`dict` cost function mapping transitions to the synchronous product net
    skip: :class:`Any` symbol to use for skips in the alignment

    Returns
    -------
    dictionary : :class:`dict` with keys **alignment**, **cost**, **visited_states**, **queued_states**
    and **traversed_arcs**
    """
    return __search(sync_prod, initial_marking, final_marking, cost_function, skip,
                    ret_tuple_as_trans_desc=ret_tuple_as_trans_desc, max_align_time_trace=max_align_time_trace)


def __search(sync_net, ini, fin, cost_function, skip, ret_tuple_as_trans_desc=False,
             max_align_time_trace=sys.maxsize):
    start_time = time.time()

    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    incidence_matrix = inc_mat_construct(sync_net)
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)

    closed = set()
    reach_fm = False
    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    g_matrix = -np.eye(len(sync_net.transitions))
    h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()
    cost_vec = [x * 1.0 for x in cost_vec]
    a_matrix = matrix(a_matrix)
    g_matrix = matrix(g_matrix)
    h_cvx = matrix(h_cvx)
    cost_vec = matrix(cost_vec)


    h, x = utils.__compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,ini,fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,use_cvxopt=True)
    ini_state = utils.SearchTuple(0 + h, 0, h, ini, None, None, x, True)
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1


    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    while not len(open_set) == 0:

        curr = heapq.heappop(open_set)
        current_marking = curr.m

        while not curr.trust:
            if (time.time() - start_time) > max_align_time_trace:
                return None

            already_closed = current_marking in closed
            if already_closed:

                curr = heapq.heappop(open_set)
                current_marking = curr.m
                continue

            h, x = utils.__compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix, curr.m,
                                                               fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                               use_cvxopt=True)
            lp_solved += 1

            # 11/10/19: shall not a state for which we compute the exact heuristics be
            # by nature a trusted solution?
            tp = utils.SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
            # 11/10/2019 (optimization ZA) heappushpop is slightly more efficient than pushing
            # and popping separately
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
        if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
            continue

        # 12/10/2019: do it again, since the marking could be changed
        already_closed = current_marking in closed
        if already_closed:
            continue

        # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
        # (underestimation of the remaining cost) is 0. Low-hanging fruits
        if curr.h < 0.01:
            if current_marking == fin:
                alignment, optimal_cost = __reconstruct_alignment(curr)
                other_alignments = __update_search(sync_net,
                                                    fin,
                                                    cost_function,
                                                    optimal_cost,
                                                    skip,
                                                    open_set,
                                                    closed,
                                                    a_matrix,
                                                    h_cvx,
                                                    g_matrix,
                                                    cost_vec,
                                                    incidence_matrix,
                                                    fin_vec
                                                    )
                all_alignments = merge_dict(other_alignments, alignment)
                return all_alignments, optimal_cost

        closed.add(current_marking)
        visited += 1

        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]

        for t, cost in trans_to_visit_with_cost:
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)

            if new_marking in closed:
                continue
            g = curr.g + cost

            queued += 1
            h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
            trustable = utils.__trust_solution(x)
            new_f = g + h

            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
            heapq.heappush(open_set, tp)


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def __update_search(sync_net,
                    fin,
                    cost_function,
                    optimal_cost,
                    skip,
                    open_set,
                    closed,
                    a_matrix,
                    h_cvx,
                    g_matrix,
                    cost_vec,
                    incidence_matrix,
                    fin_vec
                ):
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    other_alignments = {}

    while not len(open_set) == 0:
        curr = heapq.heappop(open_set)

        # if the cost is greater than the optimal cost, we do not continue
        if curr.f > optimal_cost + PARAM_COST_TOLERANCE:
            return other_alignments

        current_marking = curr.m

        while not curr.trust:
            already_closed = current_marking in closed
            if already_closed:
                if len(open_set) == 0:
                    return other_alignments
                curr = heapq.heappop(open_set)
                current_marking = curr.m
                continue

            h, x = utils.__compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                               incidence_matrix, curr.m,
                                                               fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                               use_cvxopt=True)
            # lp_solved += 1

            # 11/10/19: shall not a state for which we compute the exact heuristics be
            # by nature a trusted solution?
            tp = utils.SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
            # 11/10/2019 (optimization ZA) heappushpop is slightly more efficient than pushing
            # and popping separately
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
        if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
            continue

        # 12/10/2019: do it again, since the marking could be changed
        already_closed = current_marking in closed
        if already_closed:
            continue

        # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
        # (underestimation of the remaining cost) is 0. Low-hanging fruits
        if curr.h < 0.01:
            if current_marking == fin:
                other_alignment, cost = __reconstruct_alignment(curr,        ret_tuple_as_trans_desc=False)
                other_alignments = merge_dict(other_alignments, other_alignment)
                continue

        closed.add(current_marking)
        # visited += 1
        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]

        for t, cost in trans_to_visit_with_cost:
            # traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)
            if new_marking in closed:
                continue
            g = curr.g + cost
            # queued += 1
            h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
            trustable = utils.__trust_solution(x)
            new_f = g + h
            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
            heapq.heappush(open_set, tp)
    return other_alignments
#

def __reconstruct_alignment(state, ret_tuple_as_trans_desc=False):
    alignment = list()
    if state.p is not None and state.t is not None:
        parent = state.p
        if ret_tuple_as_trans_desc:
            alignment = [(state.t.name, state.t.label)]
            while parent.p is not None:
                alignment = [(parent.t.name, parent.t.label)] + alignment
                parent = parent.p
        else:
            alignment = [state.t.label]
            while parent.p is not None:
                alignment = [parent.t.label] + alignment
                parent = parent.p
    return {tuple(alignment): state.g}, state.g


@ray.remote
def compute_alignment(trace, model_path):
    net, model_im, model_fm = pnml_importer.apply(model_path)
    model_im = initial_marking.discover_initial_marking(net)
    model_fm = final_marking.discover_final_marking(net)
    return apply(trace, net, model_im, model_fm)


start_time = time.time()
ray.init(num_cpus=8)

# Import the event log and the Petri net model
log = xes_importer.apply("../log/prAm6_variants.xes")

model_path = '../model/prAm6.pnml'

# Compute the alignments in parallel
alignment_futures = [compute_alignment.remote(trace, model_path) for trace in log]
alignments = ray.get(alignment_futures)

# Output the alignments
print(time.time()-start_time)

