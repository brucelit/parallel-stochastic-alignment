import re
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

from pm4py.objects.petri_net.utils.align_utils import add_markings
from tqdm import tqdm
from src.petri_importer import import_net
from pm4py.objects.log.importer.xes import importer as xes_importer
from src.path_probability import get_path_prob
from src.tools import *
from pm4py.objects.petri_net.utils import incidence_matrix as construct_incidence_matrix


class AlignmentWithSplitAstar:
    """
    Alignment computed with split-point-based search algorithm.
    It is based on algorithm proposed in paper [1].

    Attributes:
        ini: The initial marking
        fin: The final marking
        cost_function: the cost function used
        incidence_matrix: The incidence matrix of synchronous product net
        trace_len: The number of event in the trace
        open_set: The open set used for search
        split_lst: The list of split points
        trace_sync: The list of index for sync move
        trace_log: The list of index for log move
        visited: The number of markings visited during search
        traverseds: The number of arcs traversed during search
        simple_lp: The number of regular lp solved
        complex_lp: The number of complex lp with split point list solved
        restart: The number of restart during search
        heuristic_time: The time spent on computing h-value
        queue_time: The time spent on open set related operation
        num_insert: The num of marking insertion into open set
        num_removal: The num of marking removal from open set
        num_update: The num of marking update from open set
        order: The order of marking explored
        max_rank: The max index of event explained so far

    References:
    [1] van Dongen, B.F. (2018). Efficiently Computing Alignments using extended marking equation.
    """

    def __init__(self, ini, fin, cost_function, incidence_matrix, trace_sync, trace_log):
        self.ini = ini
        self.fin = fin
        self.cost_function = cost_function
        self.incidence_matrix = incidence_matrix
        self.open_set = MinHeap()
        self.split_lst = []
        self.trace_sync = trace_sync
        self.trace_log = trace_log
        self.trace_len = len(self.trace_log)
        self.visited = 0
        self.traversed = 0
        self.simple_lp = 0
        self.complex_lp = 0
        self.restart = 0
        self.heuristic_time = 0
        self.queue_time = 0
        self.num_insert = 0
        self.num_update = 0
        self.num_removal = 0
        self.order = 0
        self.max_rank = -2

    def search(self):
        """
        Find a shortest path in the sync product net as alignment result.

        Returns
        -------
        alignment: dict
            The alignment result and other metrics recorded during search
        """

        ini_vec, fin_vec, cost_vec = vectorize_initial_final_cost(self.incidence_matrix, self.ini, self.fin,
                                                                  self.cost_function)
        closed = {}
        p_index = self.incidence_matrix.places
        inc_matrix = self.incidence_matrix.a_matrix
        cons_matrix = self.incidence_matrix.b_matrix
        start_time = timeit.default_timer()
        h, x = compute_init_heuristic_without_split(fin_vec - ini_vec, inc_matrix, cost_vec)
        self.heuristic_time += timeit.default_timer() - start_time
        self.simple_lp += 1
        # add initial marking to open set
        ini_state = NormalMarking(h, 0, h, self.ini, None, None, x, True, self.order)
        start_time = timeit.default_timer()
        self.open_set.heap_insert(ini_state)
        self.queue_time += timeit.default_timer() - start_time
        self.num_insert += 1
        # while not all states visited
        while self.open_set:
            start_time = timeit.default_timer()
            new_curr = self.open_set.heap_pop()
            self.queue_time += timeit.default_timer() - start_time
            self.num_removal += 1
            curr, flag = self._close_or_update_marking(new_curr, cost_vec, fin_vec)
            if flag == "CLOSEDSUCCESSFUL":
                closed[curr.m] = curr.g
                self.visited += 1
                self._expand_from_current_marking(curr, cost_vec, closed)
            elif flag == "REQUEUED":
                start_time = timeit.default_timer()
                self.open_set.heap_insert(curr)
                self.queue_time += timeit.default_timer() - start_time
                self.num_insert += 1
            elif flag == "RESTARTNEEDED":
                self.split_lst = sorted(self.split_lst)
                start_time = timeit.default_timer()

                h, x = compute_init_heuristic_with_split(ini_vec, fin_vec, cost_vec, self.split_lst, inc_matrix,
                                                         cons_matrix,
                                                         self.incidence_matrix.transitions, p_index,
                                                         self.trace_sync, self.trace_log)
                self.heuristic_time += timeit.default_timer() - start_time
                self.complex_lp += 1
                self.restart += 1
                # restart by reset open set and closed set
                closed = {}
                self.order = 0
                ini_state = NormalMarking(h, 0, h, self.ini, None, None, x, True, self.order)
                start_time = timeit.default_timer()
                self.open_set.heap_clear()
                self.open_set.heap_insert(ini_state)
                self.queue_time += timeit.default_timer() - start_time
                self.num_insert += 1
                self.max_rank = -2
            elif flag == "FINALMARKINGFOUND":
                return self._reconstruct_alignment(curr, self.trace_len)
            elif flag == "CLOSEDINFEASIBLE":
                closed[curr.m] = curr.g

    def _close_or_update_marking(self, marking, cost_vec, fin_vec):
        """
        Put marking in closed set if marking is feasible, otherwise compute an exact h-value for it.
        """
        if marking.m == self.fin and marking.h == 0:
            return marking, "FINALMARKINGFOUND"
        if not marking.trust:
            # compute the exact heuristics
            start_time = timeit.default_timer()
            marking_diff = fin_vec - self.incidence_matrix.encode_marking(marking.m)
            h, x, trustable, self.split_lst, self.max_rank = \
                compute_new_heuristic(marking, self.split_lst, marking_diff, self.ini,
                                      self.incidence_matrix.a_matrix, cost_vec, self.max_rank, self.trace_len)
            self.heuristic_time += timeit.default_timer() - start_time
            if h == -1:
                # need to restart
                return marking, "RESTARTNEEDED"
            # heuristic is not computable, from which final marking is unreachable
            elif trustable == "Infeasible":
                self.simple_lp += 1
                return marking, "CLOSEDINFEASIBLE"
            # if the heuristic is higher push the head of the queue down, set the score to exact score
            elif h > marking.h:
                self.simple_lp += 1
                marking.f = marking.g + h
                marking.h = h
                marking.x = x
                marking.trust = True
                # need to requeue the marking
                return marking, "REQUEUED"
            else:
                self.simple_lp += 1
                # continue with this marking
                marking.f = marking.g + h
                marking.h = h
                marking.x = x
                marking.trust = trustable
        return marking, "CLOSEDSUCCESSFUL"

    def _expand_from_current_marking(self, curr, cost_vec, closed):
        """
        Expand all subsequent markings from current marking.
        """

        # get subsequent firing transitions
        enabled_trans1 = {}
        for p in curr.m:
            for t in p.ass_trans:
                if t.sub_marking <= curr.m:
                    enabled_trans1[t] = self.incidence_matrix.transitions[t]
                    enabled_trans1 = dict(sorted(enabled_trans1.items(), key=lambda item: item[1]))
        enabled_trans = enabled_trans1.keys()
        trans_to_visit_with_cost = [(t, self.cost_function[t]) for t in enabled_trans if not (
                t is not None and is_log_move(t, '>>') and is_model_move(t, '>>'))]
        for t, cost in trans_to_visit_with_cost:
            # compute the new g score of the subsequent marking reached if t would be fired
            new_g = curr.g + cost
            new_m = add_markings(curr.m, t.add_marking)
            self.traversed += 1
            # subsequent marking is fresh, compute the f score of this path and add it to open set
            if new_m in closed:
                # the heuristic is not consistent, th us smaller g than closed could happen
                if closed[new_m] > new_g:
                    del closed[new_m]
                    self.order += 1
                    new_h, new_x, trustable = derive_heuristic(cost_vec, curr.x, self.incidence_matrix.transitions[t],
                                                               curr.h)
                    marking_to_reopen = NormalMarking(new_g + new_h, new_g, new_h, new_m, curr, t, new_x, trustable,
                                                      self.order)
                    if trustable and t.label[0] != ">>":
                        curr_max = get_max_events(marking_to_reopen)
                        if curr_max > self.max_rank:
                            self.max_rank = curr_max
                    start_time = timeit.default_timer()
                    self.open_set.heap_insert(marking_to_reopen)
                    self.queue_time += timeit.default_timer() - start_time
                    self.num_insert += 1
            else:
                start_time = timeit.default_timer()
                m_in_open = self.open_set.heap_find(new_m)
                self.queue_time += timeit.default_timer() - start_time
                # subsequent marking is already in open set
                if m_in_open:
                    start_time = timeit.default_timer()
                    marking_to_explore = self.open_set.heap_get(new_m)
                    self.queue_time += timeit.default_timer() - start_time
                    if new_g < marking_to_explore.g:
                        marking_to_explore.p, marking_to_explore.t, marking_to_explore.g = curr, t, new_g
                        marking_to_explore.h, marking_to_explore.x, marking_to_explore.trust = \
                            derive_heuristic(cost_vec, curr.x, self.incidence_matrix.transitions[t], curr.h)
                        marking_to_explore.f = marking_to_explore.g + marking_to_explore.h
                        start_time = timeit.default_timer()
                        self.open_set.heap_update(marking_to_explore)
                        self.queue_time += timeit.default_timer() - start_time
                        self.num_update += 1
                        if t.label[0] != ">>":
                            curr_max = get_max_events(marking_to_explore)
                            if curr_max > self.max_rank:
                                self.max_rank = curr_max
                    # subsequent marking has equal path, but the heuristic change from infeasible to feasible
                    elif not marking_to_explore.trust:
                        new_h, new_x, trustable = derive_heuristic(cost_vec, curr.x,
                                                                   self.incidence_matrix.transitions[t], curr.h)
                        if trustable:
                            marking_to_explore.h = new_h
                            marking_to_explore.f = new_h + marking_to_explore.g
                            marking_to_explore.trustable = True
                            marking_to_explore.x = new_x
                            start_time = timeit.default_timer()
                            self.open_set.heap_update(marking_to_explore)
                            self.queue_time += timeit.default_timer() - start_time
                            self.num_update += 1
                            if t.label[0] != ">>":
                                curr_max = get_max_events(marking_to_explore)
                                if curr_max > self.max_rank:
                                    self.max_rank = curr_max
                else:
                    self.order += 1
                    new_h, new_x, trustable = derive_heuristic(cost_vec, curr.x, self.incidence_matrix.transitions[t],
                                                               curr.h)
                    new_marking_to_explore = NormalMarking(new_g + new_h, new_g, new_h, new_m, curr, t, new_x,
                                                           trustable,
                                                           self.order)
                    if trustable and t.label[0] != ">>":
                        curr_max = get_max_events(new_marking_to_explore)
                        if curr_max > self.max_rank:
                            self.max_rank = curr_max
                    start_time = timeit.default_timer()
                    self.open_set.heap_insert(new_marking_to_explore)
                    self.queue_time += timeit.default_timer() - start_time
                    self.num_insert += 1

    def _reconstruct_alignment(self, state, ret_tuple_as_trans_desc=False):
        alignment = list()
        path = list()
        probability = 1
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
            'probability': probability,
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
    inc_matrix = construct_incidence_matrix.construct(model_net)

    # iterate every case in this xes log file
    for case_index in tqdm(range(len(event_log))):
        events_lst = []
        for event in event_log[case_index]:
            events_lst.append(event['concept:name'])
        trace_str = ''.join(events_lst)
        if trace_str not in trace_variant_lst:
            # construct synchronous product net
            sync_product = SynchronousProduct(event_log[case_index], model_net, model_im, model_fm)
            initial_marking, final_marking, cost_function, incidence_matrix, trace_sync, trace_log \
                = sync_product.construct_sync_product(event_log[case_index], model_net, model_im, model_fm)
            # compute alignment with regular a-star algorithm
            start_time = timeit.default_timer()
            ali_with_regular_astar = AlignmentWithSplitAstar(initial_marking, final_marking, cost_function,
                                                             incidence_matrix, trace_sync, trace_log)
            alignment_result = ali_with_regular_astar.search()
            alignment_result['total'] = timeit.default_timer() - start_time
            # alignment_result['case_id'] = event_log[case_index].attributes['concept:name']
            alignment_result['trace_length'] = len(events_lst)
            trace_variant_lst[trace_str] = alignment_result
        else:
            alignment_result = trace_variant_lst[trace_str]
            alignment_result['case_id'] = event_log[case_index].attributes['concept:name']

        # get the probability of the path
        alignment_result['probability'] = get_path_prob(alignment_result['path'], model_net, inc_matrix, model_im,
                                                        trans2weight)

        temp_df = pd.DataFrame([alignment_result], columns=df.columns)
        df = pd.concat([df, temp_df], ignore_index=True)
        # The name of result csv file is of the form: 'log_name + model_name + algorithm type.csv'
    df.to_csv('../results/log=' + log_name + '&model=' + model_name + '&algorithm=non_stochastic_split_astar' + '.csv',
              index=False)


if __name__ == '__main__':
    log_path = "../log/request_variants.xes"
    model_path = '../model/request_id005_uemsc.pnml'
    # compute alignments with regular astar
    compute_alignment(log_path, model_path)
