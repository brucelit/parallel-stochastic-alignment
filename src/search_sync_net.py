from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.objects.petri_net import properties


SKIP = ">>"
def copy_into(source_net, target_net, upper, skip):
    t_map = {}
    p_map = {}
    for t in source_net.transitions:
        name = (t.name, skip) if upper else (skip, t.name)
        label = (t.label, skip) if upper else (skip, t.label)
        t_map[t] = PetriNet.Transition(name, label)
        if properties.TRACE_NET_TRANS_INDEX in t.properties:
            # 16/02/2021: copy the index property from the transition of the trace net
            t_map[t].properties[properties.TRACE_NET_TRANS_INDEX] = t.properties[properties.TRACE_NET_TRANS_INDEX]
        target_net.transitions.add(t_map[t])

    for p in source_net.places:
        name = (p.name, skip) if upper else (skip, p.name)
        p_map[p] = PetriNet.Place(name)
        if properties.TRACE_NET_PLACE_INDEX in p.properties:
            # 16/02/2021: copy the index property from the place of the trace net
            p_map[p].properties[properties.TRACE_NET_PLACE_INDEX] = p.properties[properties.TRACE_NET_PLACE_INDEX]
        target_net.places.add(p_map[p])

    for t in source_net.transitions:
        for a in t.in_arcs:
            add_arc_from_to(p_map[a.source], t_map[t], target_net)
        for a in t.out_arcs:
            add_arc_from_to(t_map[t], p_map[a.target], target_net)

    return t_map, p_map


def is_model_move(t):
    return t.label[0] == SKIP and t.label[1] != SKIP


def is_log_move(t):
    return t.label[0] != SKIP and t.label[1] == SKIP