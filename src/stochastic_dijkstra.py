'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import heapq
import time
import sys
import pandas as pd
import copy

from pathlib import Path
from pm4py.objects.log import obj as log_implementation
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.objects.petri_net.utils.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.objects.petri_net.utils import align_utils as utils, final_marking
from pm4py.util import exec_utils
from enum import Enum
from copy import copy, deepcopy
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util import variants_util
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing
from tqdm import tqdm
from src.petri_importer import import_net
from pm4py.objects.petri_net.utils import incidence_matrix as construct_incidence_matrix
from src.tools import StochasticDijkstraSearchTuple
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer


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


def convert_paths2alignments(all_paths, trans_dict):

    optimal_alignments = []
    for each_path in all_paths:
        optimal_alignment = []
        for i in range(len(each_path)):
            optimal_alignment.append(trans_dict[each_path[i]].label)
        optimal_alignments.append(optimal_alignment)
    return optimal_alignments


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

    best_worst = apply(trace, petri_net, initial_marking, final_marking, parameters=parameters)

    if best_worst is not None:
        return best_worst['cost']

    return None


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

    alignment = apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, parameters)
    return alignment


def apply_from_variant(variant, petri_net, initial_marking, final_marking, parameters=None):
    """
    Apply the alignments from the specification of a single variant

    Parameters
    -------------
    variant
        Variant (as string delimited by the "variant_delimiter" parameter)
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm (same as 'apply' method, plus 'variant_delimiter' that is , by default)

    Returns
    ------------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    if parameters is None:
        parameters = {}
    trace = variants_util.variant_to_trace(variant, parameters=parameters)

    return apply(trace, petri_net, initial_marking, final_marking, parameters=parameters)


def apply_from_variants_dictionary(var_dictio, petri_net, initial_marking, final_marking, parameters=None):
    if parameters is None:
        parameters = {}
    dictio_alignments = {}
    for variant in var_dictio:
        dictio_alignments[variant] = apply_from_variant(variant, petri_net, initial_marking, final_marking,
                                                        parameters=parameters)
    return dictio_alignments


def apply_from_variants_list(var_list, petri_net, initial_marking, final_marking, parameters=None):
    """
    Apply the alignments from the specification of a list of variants in the log

    Parameters
    -------------
    var_list
        List of variants (for each item, the first entry is the variant itself, the second entry may be the number of cases)
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm (same as 'apply' method, plus 'variant_delimiter' that is , by default)

    Returns
    --------------
    dictio_alignments
        Dictionary that assigns to each variant its alignment
    """
    if parameters is None:
        parameters = {}
    start_time = time.time()
    max_align_time = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME, parameters,
                                                sys.maxsize)
    max_align_time_trace = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                      sys.maxsize)
    dictio_alignments = {}
    for varitem in var_list:
        this_max_align_time = min(max_align_time_trace, (max_align_time - (time.time() - start_time)) * 0.5)
        variant = varitem[0]
        parameters[Parameters.PARAM_MAX_ALIGN_TIME_TRACE] = this_max_align_time
        dictio_alignments[variant] = apply_from_variant(variant, petri_net, initial_marking, final_marking,
                                                        parameters=parameters)
    return dictio_alignments


def apply_from_variants_list_petri_string(var_list, petri_net_string, parameters=None):
    if parameters is None:
        parameters = {}

    from pm4py.objects.petri_net.importer.variants import pnml as petri_importer

    petri_net, initial_marking, final_marking = petri_importer.import_petri_from_string(petri_net_string)

    res = apply_from_variants_list(var_list, petri_net, initial_marking, final_marking, parameters=parameters)
    return res


def apply_from_variants_list_petri_string_mprocessing(mp_output, var_list, petri_net_string, parameters=None):
    if parameters is None:
        parameters = {}

    res = apply_from_variants_list_petri_string(var_list, petri_net_string, parameters=parameters)
    mp_output.put(res)


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

    return apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function,
                           utils.SKIP, ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                           max_align_time_trace=max_align_time_trace)


def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, skip, ret_tuple_as_trans_desc=False,
                    max_align_time_trace=sys.maxsize):
    return __search(sync_prod,
                    initial_marking,
                    final_marking, cost_function,
                    skip,
                    ret_tuple_as_trans_desc=ret_tuple_as_trans_desc, max_align_time_trace=max_align_time_trace)


def __search(sync_net,
             ini,
             fin,
             cost_function,
             skip,
             ret_tuple_as_trans_desc=False,
             max_align_time_trace=sys.maxsize):
    incidence_matrix = construct_incidence_matrix.construct(sync_net)
    reversed_transitions_dict = {value: key for key, value in incidence_matrix.transitions.items()}
    start_time = time.time()
    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)
    visited_set = {}
    ini_state = StochasticDijkstraSearchTuple(0, ini, None, None, 0, [[]])
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    m_idx = 0
    while not len(open_set) == 0:
        print(m_idx, "size of open set: ", len(open_set))
        # if (time.time() - start_time) > max_align_time_trace:
        #     return None
        curr = heapq.heappop(open_set)
        current_marking = curr.m
        m_idx += 1
        # if (time.time() - start_time) > max_align_time_trace:
        #     return None

        if current_marking == fin:
            # record the g value for the state
            g_to_fin = curr.g

            # get all paths
            all_paths = __update_search(sync_net,
                                      fin,
                                      cost_function,
                                      skip,
                                      g_to_fin,
                                      open_set,
                                      visited_set,
                                      incidence_matrix,
                                      ret_tuple_as_trans_desc=False,
                                      max_align_time_trace=sys.maxsize)

            # all_paths = []
            for each_sub_path in curr.path_lst:
                all_paths.append(deepcopy(each_sub_path))

            all_optimal_alignments = convert_paths2alignments(all_paths, reversed_transitions_dict)
            # print("Num of optimal alignments:", len(all_optimal_alignments), all_paths,)
            all_paths_str = []
            for each_path in all_paths:
                path = []
                for idx in each_path:
                    for k,v in incidence_matrix.transitions.items():
                        if v == idx and not utils.__is_log_move(k, skip):
                            path.append(k.label[1])
                if str(path) not in all_paths_str:
                    all_paths_str.append(str(path))

            print("all paths str: ", len(all_paths_str), all_paths_str)

            return utils.__reconstruct_alignment(curr, visited, queued, traversed,
                                                 ret_tuple_as_trans_desc=ret_tuple_as_trans_desc)

        visited_set[current_marking] = curr
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
            if new_marking in visited_set:
                # continue
                # print("marking is already visited")
                # # see whether the cost is smaller than before
                if curr.g + cost < visited_set[new_marking].g:
                    print("smaller g happens")
                    t_idx = incidence_matrix.transitions[t]
                    copied_path_lst = deepcopy(curr.path_lst)
                    new_path_lst = get_path_lst(copied_path_lst, t_idx)
                    updated_tp = StochasticDijkstraSearchTuple(curr.g + cost,
                                                               new_marking,
                                                               curr,
                                                               t,
                                                               curr.l + 1,
                                                               new_path_lst)
                    # remove the marking in open set
                    for marking in open_set:
                        if marking.m == new_marking:
                            open_set.remove(marking)
                            break
                    heapq.heapify(open_set)
                    heapq.heappush(open_set, updated_tp)
                    del visited_set[new_marking]

                elif curr.g + cost == visited_set[new_marking].g:
                    # print("equal g happens")
                    if utils.__is_model_move(t, skip):
                        new_path_lst = deepcopy(curr.path_lst)
                    else:
                        t_idx = incidence_matrix.transitions[t]
                        copied_path_lst = deepcopy(curr.path_lst)
                        new_path_lst = get_path_lst(copied_path_lst, t_idx)

                    for marking in open_set:
                        if marking.m == new_marking:
                            for sublist in new_path_lst:
                                marking.path_lst.append(deepcopy(sublist))
                    del visited_set[new_marking]

                else:
                    continue

            # reach the marking for the first time
            queued += 1

            t_idx = incidence_matrix.transitions[t]
            copied_path_lst = deepcopy(curr.path_lst)
            new_path_lst = get_path_lst(copied_path_lst, t_idx)
            tp = StochasticDijkstraSearchTuple(curr.g + cost,
                                               new_marking,
                                               curr,
                                               t,
                                               curr.l + 1,
                                               new_path_lst)
            heapq.heappush(open_set, tp)


def __update_search(sync_net,
                  fin,
                  cost_function,
                  optimal_cost,skip,
                  open_set,
                  visited_set,
                  incidence_matrix,
                  ret_tuple_as_trans_desc,
                  max_align_time_trace):
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)
    reversed_transitions_dict = {value: key for key, value in incidence_matrix.transitions.items()}

    all_paths = []

    m_idx = 0
    while not len(open_set) == 0:
        # if (time.time() - start_time) > max_align_time_trace:
        #     return None
        curr = heapq.heappop(open_set)
        current_marking = curr.m
        m_idx += 1

        # if the cost is larger than the optimal
        if curr.f > optimal_cost:
            continue

        if current_marking == fin:
            # record the g value for the state
            for each_sub_path in curr.path_lst:
                all_paths.append(deepcopy(each_sub_path))

        visited_set[current_marking] = curr
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
            # print("transition: ",t, "new marking: ", new_marking)

            if new_marking in visited_set:
                # see whether the cost is smaller than before
                if curr.g + cost < visited_set[new_marking].g:
                    print("smaller g happens")
                    t_idx = incidence_matrix.transitions[t]
                    copied_path_lst = deepcopy(curr.path_lst)
                    new_path_lst = get_path_lst(copied_path_lst, t_idx)
                    updated_tp = StochasticDijkstraSearchTuple(curr.g + cost,
                                                               new_marking,
                                                               curr,
                                                               t,
                                                               curr.l + 1,
                                                               new_path_lst)
                    # remove the marking in open set
                    for marking in open_set:
                        if marking.m == new_marking:
                            open_set.remove(marking)
                            break
                    heapq.heapify(open_set)
                    heapq.heappush(open_set, updated_tp)

                elif curr.g + cost == visited_set[new_marking].g:
                    if utils.__is_model_move(t, skip):
                        new_path_lst = deepcopy(curr.path_lst)
                    else:
                        t_idx = incidence_matrix.transitions[t]
                        copied_path_lst = deepcopy(curr.path_lst)
                        new_path_lst = get_path_lst(copied_path_lst, t_idx)

                    for marking in open_set:
                        if marking.m == new_marking:
                            for sublist in new_path_lst:
                                marking.path_lst.append(deepcopy(sublist))
                continue

            # reach the marking for the first time
            # queued += 1
            # if utils.__is_model_move(t, skip):
            #     new_path_lst = deepcopy(curr.path_lst)
            # else:
            t_idx = incidence_matrix.transitions[t]
            copied_path_lst = deepcopy(curr.path_lst)
            new_path_lst = get_path_lst(copied_path_lst, t_idx)
            tp = StochasticDijkstraSearchTuple(curr.g + cost,
                                               new_marking,
                                               curr,
                                               t,
                                               curr.l + 1,
                                               new_path_lst)
            heapq.heappush(open_set, tp)
    return all_paths


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
    model_net, model_im, model_fm = pnml_importer.apply(pnml_file)

    model_fm = final_marking.discover_final_marking(model_net)
    print("final marking: ", model_fm)
    log_name = Path(log_path).stem
    model_name = Path(model_path).stem

    # the column name in result csv file
    field_names = ['probability',
                   'cost',
                   'path',
                   # 'alignment',
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
        # if case_index > 0:
        #     break
        events_lst = []
        for event in event_log[case_index]:
            events_lst.append(event['concept:name'])
        trace_str = ', '.join(events_lst)

        if trace_str not in trace_variant_lst:
            print("case index: ", case_index, " ", trace_str)
            trace_variant_lst[trace_str] = apply(event_log[case_index], model_net, model_im, model_fm)
            alignment_result = trace_variant_lst[trace_str]

        else:
            alignment_result = trace_variant_lst[trace_str]
            alignment_result['case_id'] = event_log[case_index].attributes['concept:name']

        # get the probability of the path
        # alignment_result['probability'] = get_path_prob(alignment_result['path'], model_net, inc_matrix, model_im, trans2weight)
        # print(alignment_result)

        # temp_df = pd.DataFrame([alignment_result], columns=df.columns)
        # df = pd.concat([df, temp_df], ignore_index=True)
        # The name of result csv file is of the form: 'log_name + model_name + algorithm type.csv'
    # df.to_csv('../results/log=' + log_name + '&model=' + model_name + '&algorithm=non_stochastic_astar' + '.csv', index=False)


def get_path_lst(path_lst, t_idx):
    """
    Get the Parikh vector from previous marking.
    """
    copied_path_lst = deepcopy(path_lst)
    for each_path_lst in copied_path_lst:
        each_path_lst.append(t_idx)
    return copied_path_lst


if __name__ == '__main__':
    log_path = "../log/bpi17_sample.xes"
    model_path = '../model/bpi17_fodina.pnml'
    # compute alignments with regular astar
    compute_alignment(log_path, model_path)
