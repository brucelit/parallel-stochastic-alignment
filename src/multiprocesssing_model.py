import heapq
import logging
import multiprocessing
import multiprocessing as mp
import time
from copy import copy

import numpy as np
import pandas as pd
from cvxopt import matrix
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net import properties
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import final_marking, initial_marking
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.petri_utils import decorate_places_preset_trans, \
    decorate_transitions_prepostset, add_arc_from_to
from pm4py.util.lp import solver as lp_solver
from src.decompose_petri import decompose_into_k_subnet
from src.search_heuristic import trust_solution, compute_exact_heuristic_new_version, derive_heuristic
from src.search_marking import SearchMarking
from src.search_sync_net import copy_into, is_log_move, is_model_move

# Set the logging level to WARNING to suppress informational messages
logging.getLogger('pm4py').setLevel(logging.WARNING)

PARAM_CPU_NUM_LST = [3, 4, 5, 6, 7, 8]
PARAM_COST_TOLERANCE = [2000]
SKIP = '>>'
STD_MODEL_LOG_MOVE_COST = 1000
STD_TAU_COST = 1
STD_SYNC_COST = 0


def compute_alignments(trace_idx: int,
                       optimal_alignment_lst,
                       cost_tolerance: int,
                       trace: Trace,
                       petri_net: PetriNet,
                       petri_im: Marking,
                       petri_fm: Marking):
    # set the cost of model and log move
    model_cost_function, sync_cost_function = get_cost_map(petri_net)

    # get trace net
    trace_net, trace_im, trace_fm = construct_trace_net_cost_aware(trace)

    # construct the synchronous product net with cost
    sync_net, sync_im, sync_fm, cost_function = construct_cost_aware(trace_net,
                                                                     trace_im,
                                                                     trace_fm,
                                                                     petri_net,
                                                                     petri_im,
                                                                     petri_fm,
                                                                     model_cost_function)

    # determine whether the final marking is reached for the first time
    reach_fm = False

    # ---- initialize ------
    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)
    incidence_matrix = inc_mat_construct(sync_net)
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    fin_vec = incidence_matrix.encode_marking(sync_fm)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[incidence_matrix.transitions[t]] = 1.0 * cost_function[t]
    cost_vec = matrix(cost_vec)

    a_matrix = matrix(np.asmatrix(incidence_matrix.a_matrix).astype(np.float64))
    h_cvx = matrix(np.matrix(np.zeros(len(sync_net.transitions))).transpose())
    g_matrix = matrix(-np.eye(len(sync_net.transitions)))

    h, x = compute_exact_heuristic_new_version(sync_net,
                                               a_matrix,
                                               h_cvx,
                                               g_matrix,
                                               cost_vec,
                                               incidence_matrix,
                                               sync_im,
                                               fin_vec,
                                               lp_solver.DEFAULT_LP_SOLVER_VARIANT)
    ini_search_marking = SearchMarking(0 + h, 0, h, sync_im, None, None, x, True)
    open_set = [ini_search_marking]
    heapq.heapify(open_set)
    # visited = 0
    # queued = 0
    # traversed = 0
    # lp_solved = 1
    closed = set()
    all_alignments = []
    # ---- initialize ------

    while not len(open_set) == 0:
        curr = heapq.heappop(open_set)

        if curr.f > optimal_alignment_lst[trace_idx] + cost_tolerance:
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

            h, x = compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                                       curr.m, fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT)
            # lp_solved += 1
            tp = SearchMarking(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        # max allowed heuristics value (27/10/2019, due to the numerical instability of pm4py solvers)
        if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
            continue

        # 12/10/2019: do it again, since the marking could be changed
        already_closed = current_marking in closed
        if already_closed:
            continue

        # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
        # (underestimation of the remaining cost) is 0. Low-hanging fruits
        if curr.h < 0.01:
            if current_marking == sync_fm:
                if not reach_fm:
                    reach_fm = True
                    # reach the final marking for the first time, construct the optimal alignment
                    alignment, optimal_cost = reconstruct_optimal_alignment(curr)
                    if optimal_cost < optimal_alignment_lst[trace_idx]:
                        optimal_alignment_lst[trace_idx] = optimal_cost
                else:
                    alignment = reconstruct_alignment(curr)
                all_alignments.append(alignment)
                continue

        closed.add(current_marking)
        # visited += 1
        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and is_log_move(t) and is_model_move(t))]

        for t, cost in trans_to_visit_with_cost:
            # traversed += 1
            new_marking = add_markings(current_marking, t.add_marking)

            # no longer use if the marking is in closed set
            if new_marking in closed:
                continue
            g = curr.g + cost
            # queued += 1
            h, x = derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
            trustable = trust_solution(x)
            new_f = g + h
            tp = SearchMarking(new_f, g, h, new_marking, curr, t, x, trustable)
            heapq.heappush(open_set, tp)

    return all_alignments


def reconstruct_optimal_alignment(search_marking):
    alignment = list()
    path = list()
    if search_marking.p is not None and search_marking.t is not None:
        if not is_log_move(search_marking.t):
            path = [search_marking.t.name[1]]
        parent = search_marking.p
        alignment = [(search_marking.t.label)]
        while parent.p is not None:
            if not is_log_move(parent.t):
                path = [parent.t.name[1]] + path
            alignment = [(parent.t.label)] + alignment
            parent = parent.p
    return {tuple(alignment): search_marking.g}, search_marking.g


def reconstruct_alignment(search_marking):
    alignment = list()
    if search_marking.p is not None and search_marking.t is not None:
        parent = search_marking.p
        alignment = [(search_marking.t.label)]
        while parent.p is not None:
            alignment = [(parent.t.label)] + alignment
            parent = parent.p
    return {tuple(alignment): search_marking.g}


def construct_trace_net_cost_aware(trace):
    """
    Creates a trace net, i.e. a trace in Petri net form mapping specific costs to transitions.

    Parameters
    ----------
    trace: :class:`list` input trace, assumed to be a list of events
    costs: :class:`list` list of costs, length should be equal to the length of the input trace
    trace_name_key: :class:`str` key of the attribute that defines the name of the trace
    activity_key: :class:`str` key of the attribute of the events that defines the activity name

    Returns
    -------
    tuple: :class:`tuple` of the net, initial marking, final marking and map of costs
    """
    net = PetriNet("")
    place_map = {0: PetriNet.Place('p_0')}
    net.places.add(place_map[0])
    for i in range(0, len(trace)):
        t = PetriNet.Transition(trace[i]['concept:name'] + '_' + str(i), trace[i]['concept:name'])
        t.properties[properties.TRACE_NET_TRANS_INDEX] = i
        net.transitions.add(t)
        place_map[i + 1] = PetriNet.Place('p_' + str(i + 1))
        place_map[i + 1].properties[properties.TRACE_NET_PLACE_INDEX] = i + 1
        net.places.add(place_map[i + 1])
        add_arc_from_to(place_map[i], t, net)
        add_arc_from_to(t, place_map[i + 1], net)
    return net, Marking({place_map[0]: 1}), Marking({place_map[len(trace)]: 1})


def construct_cost_aware(pn1, im1, fm1, pn2, im2, fm2, model_costs):
    """
    Constructs the synchronous product net of two given Petri nets.

    :param pn1: Petri net 1
    :param im1: Initial marking of Petri net 1
    :param fm1: Final marking of Petri net 1
    :param pn2: Petri net 2
    :param im2: Initial marking of Petri net 2
    :param fm2: Final marking of Petri net 2
    :param skip: Symbol to be used as skip
    :param pn2_costs: dictionary mapping transitions of pn2 to corresponding costs
    :param pn1_costs: dictionary mapping pairs of transitions in pn1 and pn2 to costs
    :param sync_costs: Costs of sync moves

    Returns
    -------
    :return: Synchronous product net and associated marking labels are of the form (a,>>)
    """
    sync_net = PetriNet('synchronous_product_net of %s and %s' % (pn1.name, pn2.name))
    t1_map, p1_map = copy_into(pn1, sync_net, True, SKIP)
    t2_map, p2_map = copy_into(pn2, sync_net, False, SKIP)
    costs = dict()

    for t1 in pn1.transitions:
        costs[t1_map[t1]] = 1000
    for t2 in pn2.transitions:
        costs[t2_map[t2]] = model_costs[t2]

    for t1 in pn1.transitions:
        for t2 in pn2.transitions:
            if t1.label == t2.label:
                sync = PetriNet.Transition((t1.name, t2.name), (t1.label, t2.label))
                sync_net.transitions.add(sync)
                costs[sync] = STD_SYNC_COST
                # copy the properties of the transitions inside the transition of the sync net
                for p1 in t1.properties:
                    sync.properties[p1] = t1.properties[p1]
                for p2 in t2.properties:
                    sync.properties[p2] = t2.properties[p2]
                for a in t1.in_arcs:
                    add_arc_from_to(p1_map[a.source], sync, sync_net)
                for a in t2.in_arcs:
                    add_arc_from_to(p2_map[a.source], sync, sync_net)
                for a in t1.out_arcs:
                    add_arc_from_to(sync, p1_map[a.target], sync_net)
                for a in t2.out_arcs:
                    add_arc_from_to(sync, p2_map[a.target], sync_net)

    sync_im = Marking()
    sync_fm = Marking()
    for p in im1:
        sync_im[p1_map[p]] = im1[p]
    for p in im2:
        sync_im[p2_map[p]] = im2[p]
    for p in fm1:
        sync_fm[p1_map[p]] = fm1[p]
    for p in fm2:
        sync_fm[p2_map[p]] = fm2[p]
    sync_net.properties[properties.IS_SYNC_NET] = True
    return sync_net, sync_im, sync_fm, costs


def add_markings(curr, add):
    m = Marking()
    for p in curr.items():
        m[p[0]] = p[1]
    for p in add.items():
        m[p[0]] += p[1]
        if m[p[0]] == 0:
            del m[p[0]]
    return m


def get_cost_map(petri_net):
    model_cost_function, sync_cost_function = dict(), dict()
    for t in petri_net.transitions:
        if t.label is not None:
            model_cost_function[t] = STD_MODEL_LOG_MOVE_COST
            sync_cost_function[t] = STD_SYNC_COST
        else:
            model_cost_function[t] = STD_TAU_COST
    return model_cost_function, sync_cost_function


def sub_process(optimal_alignment_lst,
                cost_tolerance,
                event_log,
                petri_net,
                return_dict,
                process_id):
    '''

    Parameters
    ----------
    optimal_alignment_lst: the list optimal alignment costs is saved in shared memory
    cost_tolerance: specify the amount of additional deviation we can have
    event_log:
    petri_net
    return_dict
    process_id

    Returns
    -------
    '''
    petri_fm = final_marking.discover_final_marking(petri_net)
    petri_im = initial_marking.discover_initial_marking(petri_net)
    trace2alignments_lst = []

    # iterate every case in the log
    for case_index in range(len(event_log)):
        # set the cost of model and log move
        model_cost_function, sync_cost_function = get_cost_map(petri_net)

        # get trace net
        trace_net, trace_im, trace_fm = construct_trace_net_cost_aware(event_log[case_index])

        # construct the synchronous product net with cost
        sync_net, sync_im, sync_fm, cost_function = construct_cost_aware(trace_net,
                                                                         trace_im,
                                                                         trace_fm,
                                                                         petri_net,
                                                                         petri_im,
                                                                         petri_fm,
                                                                         model_cost_function)

        # determine whether the final marking is reached for the first time
        reach_fm = False

        # ---- initialize ------
        decorate_transitions_prepostset(sync_net)
        decorate_places_preset_trans(sync_net)
        incidence_matrix = inc_mat_construct(sync_net)
        trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        fin_vec = incidence_matrix.encode_marking(sync_fm)
        cost_vec = [0] * len(cost_function)
        for t in cost_function.keys():
            cost_vec[incidence_matrix.transitions[t]] = 1.0 * cost_function[t]
        cost_vec = matrix(cost_vec)

        a_matrix = matrix(np.asmatrix(incidence_matrix.a_matrix).astype(np.float64))
        h_cvx = matrix(np.matrix(np.zeros(len(sync_net.transitions))).transpose())
        g_matrix = matrix(-np.eye(len(sync_net.transitions)))

        h, x = compute_exact_heuristic_new_version(sync_net,
                                                   a_matrix,
                                                   h_cvx,
                                                   g_matrix,
                                                   cost_vec,
                                                   incidence_matrix,
                                                   sync_im,
                                                   fin_vec,
                                                   lp_solver.DEFAULT_LP_SOLVER_VARIANT)
        ini_search_marking = SearchMarking(0 + h, 0, h, sync_im, None, None, x, True)
        open_set = [ini_search_marking]
        heapq.heapify(open_set)
        # visited = 0
        # queued = 0
        # traversed = 0
        # lp_solved = 1
        closed = set()
        all_alignments = []
        # ---- initialize ------

        while not len(open_set) == 0:
            curr = heapq.heappop(open_set)

            if curr.f > optimal_alignment_lst[case_index] + cost_tolerance:
                trace2alignments_lst.append(all_alignments)
                break

            current_marking = curr.m

            while not curr.trust:
                already_closed = current_marking in closed

                if already_closed:
                    if len(open_set) == 0:
                        trace2alignments_lst.append(all_alignments)
                        break

                    curr = heapq.heappop(open_set)
                    current_marking = curr.m
                    continue

                h, x = compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                           incidence_matrix,
                                                           curr.m, fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT)
                # lp_solved += 1
                tp = SearchMarking(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
                curr = heapq.heappushpop(open_set, tp)
                current_marking = curr.m

            # max allowed heuristics value (27/10/2019, due to the numerical instability of pm4py solvers)
            if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
                continue

            # 12/10/2019: do it again, since the marking could be changed
            already_closed = current_marking in closed
            if already_closed:
                continue

            # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
            # (underestimation of the remaining cost) is 0. Low-hanging fruits
            if curr.h < 0.01:
                if current_marking == sync_fm:
                    if not reach_fm:
                        reach_fm = True
                        # reach the final marking for the first time, construct the optimal alignment
                        alignment, optimal_cost = reconstruct_optimal_alignment(curr)
                        if optimal_cost < optimal_alignment_lst[case_index]:
                            optimal_alignment_lst[case_index] = optimal_cost
                            print("case index: ", case_index, " for net: ", len(petri_net.transitions),  "alignment: ", alignment)
                    else:
                        alignment = reconstruct_alignment(curr)
                    all_alignments.append(alignment)
                    continue

            closed.add(current_marking)
            # visited += 1
            enabled_trans = copy(trans_empty_preset)
            for p in current_marking:
                for t in p.ass_trans:
                    if t.sub_marking <= current_marking:
                        enabled_trans.add(t)

            trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                    t is not None and is_log_move(t) and is_model_move(t))]

            for t, cost in trans_to_visit_with_cost:
                # traversed += 1
                new_marking = add_markings(current_marking, t.add_marking)

                # no longer use if the marking is in closed set
                if new_marking in closed:
                    continue
                g = curr.g + cost
                # queued += 1
                h, x = derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                trustable = trust_solution(x)
                new_f = g + h
                tp = SearchMarking(new_f, g, h, new_marking, curr, t, x, trustable)
                heapq.heappush(open_set, tp)
    return_dict[process_id] = trace2alignments_lst


if __name__ == '__main__':
    # import the event log
    log_path = "../log/prAm6_variants_100.xes"
    event_log = xes_importer.apply(log_path)

    # import the petri net
    net, im, fm = pnml_importer.apply("../model/prAm6_fodina.pnml")
    print("original net transition: ", len(net.transitions), " places: ", len(net.places))

    time_result = []

    for cost_tolerance in PARAM_COST_TOLERANCE:
        temp_time_result = []

        for cpu_num in PARAM_CPU_NUM_LST:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            start_time = time.time()

            overlap_threshold = 0.5
            net_lst = decompose_into_k_subnet(net, cpu_num, overlap_threshold)
            optimal_alignment_lst = mp.Array('i', [1000000 for i in range(len(event_log))])

            processes = []
            if cpu_num > len(net_lst):
                print("trace num smaller than cpu num")
                break

            process_id = 0
            for each_net in net_lst:
                print("each net transition: ", len(each_net.transitions), " places: ", len(each_net.places))

                # Create and start a process for each task
                p = multiprocessing.Process(target=sub_process,
                                            args=(optimal_alignment_lst,
                                                  cost_tolerance,
                                                  event_log,
                                                  each_net,
                                                  return_dict,
                                                  process_id))
                processes.append(p)
                p.start()
                process_id += 1

            # Wait for all processes to complete
            for p in processes:
                p.join()

            all_alignment_result = {key: [] for key in range(len(event_log))}

            # for each net, get the list of alignments list
            for net_id, alignments_lst_lst in return_dict.items():
                for alignments_lst_idx in range(len(alignments_lst_lst)):
                    for each_alignment in alignments_lst_lst[alignments_lst_idx]:
                        for k, v in list(each_alignment.items()):
                            if v > optimal_alignment_lst[alignments_lst_idx] + cost_tolerance:
                                continue
                            all_alignment_result[alignments_lst_idx].append(each_alignment)
            print("net num: ", len(net_lst), "cpu_num: ", cpu_num, "cost tolerance: ", cost_tolerance,
                  " optimal alignment lst: ", list(optimal_alignment_lst))
            print("time it takes: ", time.time() - start_time)
            temp_time_result.append(time.time() - start_time)
        time_result.append(temp_time_result)

    # result = pd.DataFrame(time_result)
    # result.to_excel('../results/PrAm6_result_multiprocess_model_fodina.xlsx')
