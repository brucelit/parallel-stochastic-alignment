import random
import warnings

import ray
import heapq
import sys
import time
import numpy as np
import logging

from cvxopt import matrix
from copy import copy
from enum import Enum
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
from ray._private.services import get_node_ip_address
from tqdm import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from cvxopt import matrix

from src.decompose_petri import decompose_into_k_subnet

# Set the logging level to WARNING to suppress informational messages
logging.getLogger('pm4py').setLevel(logging.WARNING)

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
PARAM_CPU_NUM = [4,5,6,7,8]

SKIP = '>>'
STD_MODEL_LOG_MOVE_COST = 1
STD_TAU_COST = 0.01
STD_SYNC_COST = 0

def reconstruct_alignment(state, ret_tuple_as_trans_desc=False):
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


def is_model_move(t, skip):
    return t.label[0] == skip and t.label[1] != skip


def trust_solution(x):
    for v in x:
        if v < -0.001:
            return False
    return True


def is_log_move(t, skip):
    return t.label[0] != skip and t.label[1] == skip


def derive_heuristic(incidence_matrix, cost_vec, x, t, h):
    x_prime = x.copy()
    x_prime[incidence_matrix.transitions[t]] -= 1
    return max(0, h - cost_vec[incidence_matrix.transitions[t]]), x_prime


def compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix, marking,
                                        fin_vec, variant, strict=True):
    m_vec = incidence_matrix.encode_marking(marking)
    b_term = [i - j for i, j in zip(fin_vec, m_vec)]
    b_term = np.matrix([x * 1.0 for x in b_term]).transpose()

    if not strict:
        g_matrix = np.vstack([g_matrix, a_matrix])
        h_cvx = np.vstack([h_cvx, b_term])
        a_matrix = np.zeros((0, a_matrix.shape[1]))
        b_term = np.zeros((0, b_term.shape[1]))

    b_term = matrix(b_term)
    parameters_solving = {"solver": "glpk"}

    sol = lp_solver.apply(cost_vec, g_matrix, h_cvx, a_matrix, b_term, parameters=parameters_solving, variant=variant)
    prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
    points = lp_solver.get_points_from_sol(sol, variant=variant)

    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = points if points is not None else [0.0] * len(sync_net.transitions)

    return prim_obj, points


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def add_markings(curr, add):
    m = Marking()
    for p in curr.items():
        m[p[0]] = p[1]
    for p in add.items():
        m[p[0]] += p[1]
        if m[p[0]] == 0:
            del m[p[0]]
    return m


def vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function):
    ini_vec = incidence_matrix.encode_marking(ini)
    fini_vec = incidence_matrix.encode_marking(fin)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[incidence_matrix.transitions[t]] = cost_function[t]
    return ini_vec, fini_vec, cost_vec

class SearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif other.f < self.f:
            return False
        elif self.trust and not other.trust:
            return True
        else:
            return self.h < other.h

    def __get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.__get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " f=" + str(self.f), ' g=' + str(self.g), " h=" + str(self.h),
                        " path=" + str(self.__get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)


def initiate_alignment_cost(log_len):
    return [1000000000 for i in range(log_len)]


@ray.remote
class Coordinator:
    def __init__(self, log_len):
        self.alignments = self.initiate_alignment_dict(log_len)
        self.target_num = initiate_alignment_cost(log_len)
        self.worker_handles = []

    def get_target_num(self):
        return self.target_num

    def initiate_alignment_dict(self, log_len):
        main_dict = {}
        # Populate the dictionary with keys ranging from 0 to log length
        for i in range(log_len):
            main_dict[i] = {}
        # Print the dictionary to verify
        return main_dict

    def register_worker(self, worker_handle):
        self.worker_handles.append(worker_handle)

    # def update_cost(self, optimal_cost, trace_idx):
    #     self.target_num[trace_idx] = optimal_cost
    #     # update the alignment list
    #     for k, v in list(self.alignments[trace_idx].items()):
    #         if v > optimal_cost + PARAM_COST_TOLERANCE:
    #             del self.alignments[trace_idx][k]


    def update_cost(self, optimal_cost, trace_idx, net_idx):
        if optimal_cost < self.target_num[trace_idx]:
            print("update optimal cost: ", optimal_cost, " trace id: ", trace_idx," net idx: ", net_idx)
            self.target_num[trace_idx] = optimal_cost
            self.broadcast_optimal_cost(trace_idx)


    def update_alignment(self, alignments_to_add, trace_idx):
        self.alignments[trace_idx] = merge_dict(self.alignments[trace_idx], alignments_to_add)

    def get_all_cost(self):
        return self.target_num

    def get_all_alignment(self):
        return self.alignments

    def broadcast_optimal_cost(self, trace_idx):
        print("broadcast the optimal cost result for: ", trace_idx)
        for worker in self.worker_handles:
            value = ray.get(worker.target_num.remote())

@ray.remote
class Worker:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.log = None
        self.optimal_alignment_lst = initiate_alignment_cost(log_len)


    def receive_update(self, trace_idx):
        print("receive optimal cost")
        self.optimal_alignment_lst[trace_idx] = ray.get(self.coordinator.target_num[trace_idx])


    def apply(self,
              trace_idx: int,
              net_idx: int,
              trace: Trace,
              petri_net: PetriNet,
              initial_marking: Marking,
              final_marking: Marking,
              parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.AlignmentResult:
        """
            Performs the basic alignment search, given a trace and a net.
    
            Parameters
            ----------
            trace_idx
            net_idx:
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
        trace_net_cost_aware_constr_function = exec_utils.get_param_value(
            Parameters.TRACE_NET_COST_AWARE_CONSTR_FUNCTION,
            parameters, construct_trace_net_cost_aware)

        if trace_cost_function is None:
            trace_cost_function = list(
                map(lambda e: STD_MODEL_LOG_MOVE_COST, trace))
            parameters[Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

        if model_cost_function is None:
            # reset variables value
            model_cost_function = dict()
            sync_cost_function = dict()
            for t in petri_net.transitions:
                if t.label is not None:
                    model_cost_function[t] = STD_MODEL_LOG_MOVE_COST
                    sync_cost_function[t] = STD_SYNC_COST
                else:
                    model_cost_function[t] = STD_TAU_COST
            parameters[Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
            parameters[Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function

        if trace_net_constr_function is not None:
            # keep the possibility to pass TRACE_NET_CONSTR_FUNCTION in this old version
            trace_net, trace_im, trace_fm = trace_net_constr_function(trace, activity_key=activity_key)
        else:
            trace_net, trace_im, trace_fm, parameters[
                Parameters.PARAM_TRACE_NET_COSTS] = (
                trace_net_cost_aware_constr_function(trace,
                                                     trace_cost_function,
                                                     activity_key=activity_key))

        alignment = self.apply_trace_net(trace_idx, net_idx,
                                         petri_net, initial_marking, final_marking, trace_net, trace_im,
                                         trace_fm,
                                         parameters)

        return alignment

    def apply_trace_net(self,
                        trace_idx,
                        net_idx,
                        petri_net,
                        initial_marking,
                        final_marking,
                        trace_net,
                        trace_im,
                        trace_fm,
                        parameters=None):
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
                                                                            SKIP)
            cost_function = utils.construct_standard_cost_function(sync_prod, SKIP)
        else:
            revised_sync = dict()
            for t_trace in trace_net.transitions:
                for t_model in petri_net.transitions:
                    if t_trace.label == t_model.label:
                        revised_sync[(t_trace, t_model)] = sync_cost_function[t_model]

            sync_prod, sync_initial_marking, sync_final_marking, cost_function = construct_cost_aware(
                trace_net, trace_im, trace_fm, petri_net, initial_marking, final_marking, SKIP,
                trace_net_costs, model_cost_function, revised_sync)

        max_align_time_trace = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                          sys.maxsize)

        alignment = self.apply_sync_prod(trace_idx, net_idx,
                                         sync_prod, sync_initial_marking, sync_final_marking,
                                         cost_function,
                                         SKIP)

        return_sync_cost = exec_utils.get_param_value(Parameters.RETURN_SYNC_COST_FUNCTION, parameters, False)
        if return_sync_cost:
            # needed for the decomposed alignments (switching them from state_equation_less_memory)
            return alignment, cost_function

        return alignment

    def apply_sync_prod(self,
                        trace_idx,
                        net_idx,
                        sync_prod,
                        initial_marking,
                        final_marking,
                        cost_function,
                        skip):
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
        return self.search(trace_idx,
                           net_idx,
                           sync_prod,
                           initial_marking,
                           final_marking,
                           cost_function,
                           skip)

    def search(self,
               trace_idx,
               net_idx,
               sync_net,
               ini,
               fin,
               cost_function,
               skip):
        all_alignments = {}
        reach_fm = False
        decorate_transitions_prepostset(sync_net)
        decorate_places_preset_trans(sync_net)
        incidence_matrix = inc_mat_construct(sync_net)
        ini_vec, fin_vec, cost_vec = vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)
        closed = set()

        a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
        g_matrix = -np.eye(len(sync_net.transitions))
        h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()
        cost_vec = [x * 1.0 for x in cost_vec]
        a_matrix = matrix(a_matrix)
        g_matrix = matrix(g_matrix)
        h_cvx = matrix(h_cvx)
        cost_vec = matrix(cost_vec)

        h, x = compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                   incidence_matrix,
                                                   ini,
                                                   fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT
                                                   )
        ini_state = SearchTuple(0 + h, 0, h, ini, None, None, x, True)
        open_set = [ini_state]
        heapq.heapify(open_set)
        visited = 0
        queued = 0
        traversed = 0
        lp_solved = 1

        trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        while not len(open_set) == 0:
            curr = heapq.heappop(open_set)

            if curr.f > self.optimal_alignment_lst[trace_idx] + PARAM_COST_TOLERANCE:
                return all_alignments

            current_marking = curr.m

            while not curr.trust:
                already_closed = current_marking in closed

                if already_closed:
                    if len(open_set) == 0:
                        return all_alignments

                    curr = heapq.heappop(open_set)
                    current_marking = curr.m
                    continue

                h, x = compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                           incidence_matrix, curr.m, fin_vec,
                                                           lp_solver.DEFAULT_LP_SOLVER_VARIANT)
                lp_solved += 1
                tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
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
                    alignment, cost = reconstruct_alignment(curr)
                    if cost < self.optimal_alignment_lst[trace_idx]:
                        self.coordinator.update_cost.remote(cost, trace_idx,net_idx)
                    all_alignments = merge_dict(all_alignments, alignment)

                    continue

            closed.add(current_marking)
            visited += 1

            enabled_trans = copy(trans_empty_preset)
            for p in current_marking:
                for t in p.ass_trans:
                    if t.sub_marking <= current_marking:
                        enabled_trans.add(t)

            trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                    t is not None and is_log_move(t, skip) and is_model_move(t, skip))]

            for t, cost in trans_to_visit_with_cost:
                traversed += 1
                new_marking = add_markings(current_marking, t.add_marking)

                if new_marking in closed:
                    continue
                g = curr.g + cost

                queued += 1
                h, x = derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                trustable = trust_solution(x)
                new_f = g + h

                tp = SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
                heapq.heappush(open_set, tp)

        return all_alignments

    def compute_alignment(self, sub_var):
        net = sub_var[0]
        fm = final_marking.discover_final_marking(net)
        im = initial_marking.discover_initial_marking(net)
        net_idx = sub_var[2]
        self.log = xes_importer.apply(sub_var[1])

        # iterate every case in this xes log file
        for case_index in range(len(self.log)):
            self.apply(case_index, net_idx, self.log[case_index], net, im, fm)


for cpu_num in PARAM_CPU_NUM:
    start_time = time.time()
    net, im, fm = pnml_importer.apply("../model/prAm6.pnml")

    net_lst = decompose_into_k_subnet(net,cpu_num)

    # Initialize Ray
    ray.init(num_cpus=cpu_num)
    # Get the local node IP address
    node_ip = get_node_ip_address()

    print(f"The local node IP address is: {node_ip}")
    # get the petri list
    log_path = "../log/prAm6_variants_1.xes"
    event_log = xes_importer.apply(log_path)
    log_len = len(event_log)
    coordinator = Coordinator.remote(log_len)
    # set sub task
    sub_tasks = []
    net_idx = 0
    workers = []
    for each_sub_net in net_lst:
        sub_tasks.append([each_sub_net, log_path, net_idx])
        print("net: ", net_idx," with transition: ", len(each_sub_net.transitions), " with places: ",len(each_sub_net.places))
        workers.append(Worker.remote(coordinator))
        net_idx += 1

    print("worker list: ", len(workers))
    # Process each sub-graph in parallel
    results = [worker.compute_alignment.remote(sub_task) for worker, sub_task in zip(workers, sub_tasks)]

    # Collect the results
    final_results = ray.get(results)

    # Print the final results
    optimal_cost = ray.get(coordinator.get_all_cost.remote())
    optimal_alignments = ray.get(coordinator.get_all_alignment.remote())

    # Get the latest result from the coordinator
    # latest_result = ray.get(coordinator.get_latest_result.remote())

    # Shut down Ray
    ray.shutdown()
    print("optimal cost: ", optimal_cost)
    print("running time: ", time.time() - start_time)
