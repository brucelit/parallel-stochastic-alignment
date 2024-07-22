from copy import copy,deepcopy

import pm4py
from pm4py import Marking

from pm4py.objects.petri_net.utils import align_utils as utils
from src.tools import decorate_transitions_prepostset, decorate_places_preset_trans


def get_path_prob(path, pn, incidence_matrix, im, trans2weight):
    """
    Get the probability of a path.
    :param path: list of transitions
    :param pn: PetriNet
    :param im: InitialMarking
    :return: float
    """

    decorate_transitions_prepostset(pn)
    decorate_places_preset_trans(pn)
    current_marking = im
    trans_empty_preset = set(t for t in incidence_matrix.transitions if len(t.in_arcs) == 0)

    # start from initial marking, fire the transition
    i = 0
    path_prob = 1

    while i < len(path):
        probability_sum = 0
        new_marking = Marking()
        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)
                    probability_sum += trans2weight[t.name]
        for t in enabled_trans:
            if t.name == path[i]:
                path_prob = trans2weight[t.name]/probability_sum * path_prob
                new_marking = utils.add_markings(current_marking, t.add_marking)
                i += 1
                break
        current_marking = new_marking
    return path_prob