"""
The code is directly based on the one implemented in PM4Py, with changes on lp solver used for heuristic computation.

References:
https://github.com/pm4py/pm4py-core/blob/release/pm4py/algo/conformance/alignments/petri_net/variants/state_equation_a_star.py

"""

import heapq
import timeit
from pathlib import Path

import numpy as np

from copy import copy

import pandas as pd
from pm4py.objects.petri_net.utils import align_utils as utils
from tqdm import tqdm

from src.search_probability import get_path_prob
from src.petri_importer import import_net
from pm4py.objects.log.importer.xes import importer as xes_importer

from src.stochastic_searchtuple import SearchTuple
from src.tools import derive_heuristic, compute_init_heuristic_without_split, is_log_move, is_model_move, \
    SynchronousProduct
from pm4py.objects.petri_net.utils import incidence_matrix as construct_incidence_matrix


class AlignmentWithRegularAstar:
    """
    Alignment computed with regular A* algorithm.
    It is based on regular A* algorithm proposed in paper [1].

    Attributes:
        ini: The initial marking
        fin: The final marking
        cost_function: the cost function used
        incidence_matrix: The incidence matrix of synchronous product net
        visited: The number of markings visited during search
        queued: The number of markings queued during search
        traversed: The number of arcs traversed during search
        simple_lp: The number of regular lp solved
        heuristic_time: The time spent on computing h-value
        queue_time: The time spent on open set related operation
        num_insert: The num of insertion into open set
        num_removal: The num of removal from open set

    References:
    [1] Sebastiaan J. van Zelst et al., "Tuning Alignment Computation: An Experimental Evaluation".
    """

    def __init__(self, ini, fin, cost_function, incidence_matrix,trans2weight):
        self.ini = ini
        self.fin = fin
        self.cost_function = cost_function
        self.incidence_matrix = incidence_matrix
        self.visited = 0
        self.traversed = 0
        self.simple_lp = 0
        self.heuristic_time = 0
        self.queue_time = 0
        self.num_insert = 0
        self.num_removal = 0
        self.trans2weight = trans2weight

    def __vectorize_initial_final_cost(self, incidence_matrix, ini, fin, cost_function):
        ini_vec = incidence_matrix.encode_marking(ini)
        fin_vec = incidence_matrix.encode_marking(fin)
        cost_vec = [0] * len(cost_function)
        for t in cost_function.keys():
            cost_vec[incidence_matrix.transitions[t]] = cost_function[t]
        return np.array(ini_vec), np.array(fin_vec), np.array(cost_vec)

    def search(self):
        incidence_matrix = self.incidence_matrix
        ini_vec, fin_vec, cost_vec = self.__vectorize_initial_final_cost(self.incidence_matrix, self.ini,
                                                                         self.fin, self.cost_function)
        closed = set()
        marking2prob = {}
        cost_vec = [x * 1.0 for x in cost_vec]
        start_time = timeit.default_timer()
        h, x = compute_init_heuristic_without_split(np.array(np.array(fin_vec) - np.array(ini_vec)),
                                                    np.array(incidence_matrix.a_matrix), np.array(cost_vec))
        self.heuristic_time += timeit.default_timer() - start_time
        ini_state = SearchTuple(0 + h, 0, h, self.ini, None, None, x, True, 1)
        open_set = [ini_state]
        self.num_insert += 1
        heapq.heapify(open_set)
        self.queue_time += timeit.default_timer() - start_time
        self.simple_lp = 1
        trans_empty_preset = set(t for t in incidence_matrix.transitions if len(t.in_arcs) == 0)
        while not len(open_set) == 0:
            start_time = timeit.default_timer()
            curr = heapq.heappop(open_set)

            # get current transition prob
            curr_probability = curr.probability

            self.queue_time += timeit.default_timer() - start_time
            self.num_removal += 1
            current_marking = curr.m
            while not curr.trust:
                already_closed = current_marking in closed
                if already_closed:
                    start_time = timeit.default_timer()
                    curr = heapq.heappop(open_set)
                    self.queue_time += timeit.default_timer() - start_time
                    self.num_removal += 1
                    current_marking = curr.m
                    continue
                start_time = timeit.default_timer()
                h, x = compute_init_heuristic_without_split(
                    np.array(fin_vec) - np.array(incidence_matrix.encode_marking(curr.m)),
                    np.array(incidence_matrix.a_matrix), np.array(cost_vec))
                self.heuristic_time += timeit.default_timer() - start_time
                self.simple_lp += 1
                tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True, curr.probability)
                start_time = timeit.default_timer()
                curr = heapq.heappushpop(open_set, tp)
                self.queue_time += timeit.default_timer() - start_time
                self.num_insert += 1
                self.num_removal += 1
                current_marking = curr.m

            already_closed = current_marking in closed
            if already_closed:
                continue

            if curr.h < 0.01:
                if current_marking == self.fin:
                    return self._reconstruct_alignment(curr)

            closed.add(current_marking)
            marking2prob[current_marking] = curr_probability
            self.visited += 1

            transition_prob_sum = 0
            enabled_trans = copy(trans_empty_preset)
            for p in current_marking:
                for t in p.ass_trans:
                    if t.sub_marking <= current_marking:
                        enabled_trans.add(t)
                        if not is_log_move(t,'>>'):
                            transition_prob_sum += self.trans2weight[t.name[1]]

            # trans_to_visit_with_cost = [(t, self.cost_function[t]) for t in enabled_trans if not (
            #         t is not None and is_log_move(t, '>>') and is_model_move(t, '>>'))]
            trans_to_visit_with_cost = [(t, self.cost_function[t]) for t in enabled_trans]

            for t, cost in trans_to_visit_with_cost:

                if not is_log_move(t,'>>'):
                    trans_prob = self.trans2weight[t.name[1]]/transition_prob_sum
                else:
                    trans_prob = 1

                self.traversed += 1
                new_marking = utils.add_markings(current_marking, t.add_marking)
                if new_marking in closed:
                    if marking2prob[new_marking] < curr_probability * trans_prob:   
                        marking2prob[new_marking] = curr_probability * trans_prob
                        closed.remove(new_marking)
                    else:
                        continue
                g = curr.g + cost
                h, x, trustable = derive_heuristic(cost_vec, curr.x, incidence_matrix.transitions[t], curr.h)
                new_f = g + h
                tp = SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable,
                                 curr_probability * trans_prob)

                start_time = timeit.default_timer()
                heapq.heappush(open_set, tp)
                self.queue_time += timeit.default_timer() - start_time
                self.num_insert += 1

    def _reconstruct_alignment(self, state, ret_tuple_as_trans_desc=False):
        alignment = list()
        path = list()
        if state.p is not None and state.t is not None:

            parent = state.p
            if ret_tuple_as_trans_desc:
                alignment = [(state.t.name, state.t.label)]
                if not is_log_move(state.t, '>>'):
                    path = [state.t.name[1]]
                while parent.p is not None:
                    if not is_log_move(parent.t, '>>'):
                        path = [parent.t.name[1]] + path
                    alignment = [(parent.t.name, parent.t.label)] + alignment
                    parent = parent.p
            else:
                alignment = [state.t.label]
                if not is_log_move(state.t, '>>'):
                    path = [state.t.name[1]]
                while parent.p is not None:
                    if not is_log_move(parent.t, '>>'):
                        path = [parent.t.name[1]] + path
                    alignment = [parent.t.label] + alignment
                    parent = parent.p

        return {
            'probability': state.probability,
            'cost': state.f,
            'path': path,
            'alignment': alignment,
            'simple_lp': self.simple_lp,
            "complex_lp": 0,
            'restart': 0,
            "heuristic": self.heuristic_time,
            "queue": self.queue_time,
            "num_insert": self.num_insert,
            "num_removal": self.num_removal,
            "num_update": 0,
            'sum': self.num_insert + self.num_removal,
            'states': self.visited,
            'arcs': self.traversed,
            'split_num': 0,
            'alignment_length': len(alignment),
        }

def compute_alignment(xes_file, pnml_file):
    """
    Compute alignments for event log with given model.
    Save alignments results and all metrics during computation in a csv file.

    Parameters
    ----------
    xes_file : .xes file
               The xes file of the event log
    pnml_file : .pnml file
                The petri net model
    """

    event_log = xes_importer.apply(xes_file)
    model_net, model_im, model_fm, trans2weight = import_net(pnml_file)
    inc_matrix = construct_incidence_matrix.construct(model_net)

    log_name = Path(log_path).stem
    model_name = Path(model_path).stem

    # the column name in result csv file
    field_names = ['probability',
                   'cost',
                   'path',
                   'alignment',
                   'total',
                   'heuristic',
                   'queue',
                   'states',
                   'arcs',
                   'sum',
                   'num_insert',
                   'num_removal',
                   'num_update',
                   'simple_lp',
                   'complex_lp',
                   'restart',
                   'split_num',
                   'trace_length',
                   'alignment_length']
    df = pd.DataFrame(columns=field_names)
    trace_variant_lst = {}

    # iterate every case in this xes log file
    for case_index in tqdm(range(len(event_log))):
        events_lst = []
        for event in event_log[case_index]:
            events_lst.append(event['concept:name'])
        trace_str = ''.join(events_lst)
        if trace_str not in trace_variant_lst:
            # construct synchronous product net
            sync_product = SynchronousProduct(event_log[case_index], model_net, model_im, model_fm)
            initial_marking, final_marking, cost_function, incidence_matrix, _, _ \
                = sync_product.construct_sync_product(event_log[case_index], model_net, model_im, model_fm)
            # compute alignment with regular a-star algorithm
            start_time = timeit.default_timer()
            ali_with_regular_astar = AlignmentWithRegularAstar(initial_marking, final_marking, cost_function, incidence_matrix, trans2weight)
            alignment_result = ali_with_regular_astar.search()
            alignment_result['total'] = timeit.default_timer() - start_time
            # alignment_result['case_id'] = event_log[case_index].attributes['concept:name']
            alignment_result['trace_length'] = len(events_lst)
            trace_variant_lst[trace_str] = alignment_result
        else:
            alignment_result = trace_variant_lst[trace_str]
            alignment_result['case_id'] = event_log[case_index].attributes['concept:name']
        # get the probability of the path

        alignment_result['probability'] = get_path_prob(alignment_result['path'], model_net, inc_matrix, model_im, trans2weight)
        temp_df = pd.DataFrame([alignment_result], columns=df.columns)
        df = pd.concat([df, temp_df], ignore_index=True)
        # The name of result csv file is of the form: 'log_name + model_name + algorithm type.csv'
    df.to_csv('../results/log=' + log_name + '&model=' + model_name + '&algorithm=stochastic_astar' + '.csv', index=False)


if __name__ == '__main__':
    log_path = "../log/road_variants.xes"
    model_path = '../model/road_id02_uemsc_spn.pnml'
    # compute alignments with regular astar
    compute_alignment(log_path, model_path)
