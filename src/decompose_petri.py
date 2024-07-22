import time

from pm4py.objects.petri_net import properties
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.objects.petri_net.importer import importer as pnml_importer


def get_start_split_place(petri_net):
    '''
    Parameters
    ----------
    petri_net: the petri net to decompose

    Returns
    -------
    get the set of choice places with multiple outgoing/incoming edges
    '''

    start_places_set = set()
    merge_places_set = set()
    for place in petri_net.places:
        # if the place has multiple outgoing edges
        if len(place.out_arcs) > 1:
            # print("find start place: ",place, "  ", place.out_arcs)
            start_places_set.add(place)
        if len(place.in_arcs) > 1:
            # print("find merge place: ",place, "  ",place.in_arcs)
            merge_places_set.add(place)
    return start_places_set, merge_places_set


def is_valid_split_pair(start_place, merge_place, place_num):
    '''
    compute the valid decomposable choice place pairs. If valid, return True
    :param start_place:
    :param merge_place:
    :param place_num:
    :return:
    '''
    is_valid = True

    to_visited_place_set = [start_place]
    backward_visited_set = set()
    # conduct breadth first search from the start place start_p
    for each_place in to_visited_place_set:
        for each_out_arc in each_place.out_arcs:
            out_node = each_out_arc.target
            flag = backward_bfs(out_node, start_place, merge_place, backward_visited_set,
                                0, place_num)
            is_valid = is_valid and flag

    to_visited_place_set = [merge_place]
    forward_visited_set = set()
    # conduct breadth first search from the start place start_p
    for each_place in to_visited_place_set:
        for each_in_arc in each_place.in_arcs:
            in_node = each_in_arc.source
            flag = forward_bfs(in_node, start_place, forward_visited_set,
                               0, place_num)
            is_valid = is_valid and flag
    return is_valid


def backward_bfs(node, start_place, merge_place, visited_set, iter, depth):
    if node == start_place:
        return False

    if iter > depth:
        # print("exceed max iter", iter, depth)
        return False

    # no more outgoing edge
    if len(node.out_arcs) == 0:
        return False

    flag = True
    for each_out_arc in node.out_arcs:
        if each_out_arc.target == merge_place:
            flag = True
        else:
            if each_out_arc.target in visited_set:
                # print("is already in during backward")
                continue
            else:
                visited_set.add(each_out_arc.target)

                flag = flag and backward_bfs(each_out_arc.target,
                                             start_place,
                                             merge_place,
                                             visited_set,
                                             iter + 1, depth)
    return flag


def forward_bfs(node, start_place, forward_visited_set, iter, depth):
    if iter > depth:
        # print("exceed max iter", iter, depth)
        return False

    # no more incoming edge
    if len(node.in_arcs) == 0:
        return False

    flag = True
    for each_in_arc in node.in_arcs:
        if each_in_arc.source == start_place:
            flag = True
        else:
            if each_in_arc.source in forward_visited_set:
                # print("is already in during forward")
                continue
            else:
                forward_visited_set.add(each_in_arc.source)
                flag = flag and forward_bfs(each_in_arc.source, start_place, forward_visited_set,
                                            iter + 1, depth)
    return flag


def is_transitive_merge_pair(merge_place_before, merge_place_after, place_num):
    '''
    check two merge places, whether they are transitive
    :param merge_place_before:
    :param merge_place_after:
    :param place_num:
    :return:
    '''
    is_valid = True

    to_visited_place_set = [merge_place_before]
    backward_visited_set = set()
    # conduct breadth first search from the start place start_p
    for each_place in to_visited_place_set:
        for each_out_arc in each_place.out_arcs:
            out_node = each_out_arc.target
            flag = merge_backward_bfs(out_node, merge_place_before, merge_place_after, backward_visited_set,
                                      0, place_num)
            is_valid = is_valid and flag

    to_visited_place_set = [merge_place_after]
    forward_visited_set = set()

    # conduct breadth first search from the merge place merge_p
    for each_place in to_visited_place_set:
        for each_in_arc in each_place.in_arcs:
            in_node = each_in_arc.source
            flag = merge_forward_bfs(in_node, merge_place_before, forward_visited_set,
                                     0, place_num)
            is_valid = is_valid and flag

    return is_valid


def merge_backward_bfs(node, merge_place_before, merge_place_after, visited_set, iter, depth):
    '''
    check whether the
    :param node:
    :param merge_place_before:
    :param merge_place_after:
    :param visited_set:
    :param iter:
    :param depth:
    :return:
    '''
    if node == merge_place_before:
        return False

    if iter > depth:
        # print("exceed max iter", iter, depth)
        return False

    # no more outgoing edge
    if len(node.out_arcs) == 0:
        return False

    flag = True
    for each_out_arc in node.out_arcs:
        if each_out_arc.target == merge_place_after:
            flag = True
        else:
            if each_out_arc.target in visited_set:
                # print("is already in during backward")
                continue
            else:
                visited_set.add(each_out_arc.target)

                flag = flag and merge_backward_bfs(each_out_arc.target,
                                                   merge_place_before,
                                                   merge_place_after,
                                                   visited_set,
                                                   iter + 1, depth)
    return flag


def merge_forward_bfs(node, merge_place_before, forward_visited_set, iter, depth):
    '''

    :param node:
    :param merge_place_before:
    :param forward_visited_set:
    :param iter:
    :param depth:
    :return:
    '''

    if iter > depth:
        # print("exceed max iter", iter, depth)
        return False

    # no more incoming edge
    if len(node.in_arcs) == 0:
        return False

    flag = True
    for each_in_arc in node.in_arcs:
        if each_in_arc.source == merge_place_before:
            flag = True
        else:
            if each_in_arc.source in forward_visited_set:
                # print("is already in during forward")
                continue
            else:
                forward_visited_set.add(each_in_arc.source)
                flag = flag and merge_forward_bfs(each_in_arc.source, merge_place_before, forward_visited_set,
                                                  iter + 1, depth)
    return flag

def extract_subnet_along_outgoing_edges(original_net, start_place, target_place):
    #   get the number of transitions in the net
    total_trans_num = len(original_net.transitions)

    # get all reachable transition from start choice place to merge place
    all_reachable_transition, all_reachable_place = __find_all_reachable_transitions_dfs(start_place, target_place)

    min_overlap = 1
    transition2remove4net1 = set()
    transition2remove4net2 = set()
    place2remove4net1 = set()
    place2remove4net2 = set()
    transition2use = None

    #   split the net into two from the start place
    for each_out_arc in start_place.out_arcs:

        # the transitions to remove
        transition2keep4net1, place2keep4net1 = find_transitions_dfs(each_out_arc.target, target_place)

        transition2keep4net2 = set()
        place2keep4net2 = set()

        for each_out_arc2 in start_place.out_arcs:
            if each_out_arc != each_out_arc2:
                other_trans_set, other_place_set = find_transitions_dfs(each_out_arc2.target, target_place)
                transition2keep4net2.update(other_trans_set)
                place2keep4net2.update(other_place_set)

        common_transition_in_rg = len(transition2keep4net1.intersection(transition2keep4net2))

        # get the common transition
        common_transition = total_trans_num - len(all_reachable_transition) + common_transition_in_rg

        #         get the transition in subnet1
        total_transition_num1 = total_trans_num - len(all_reachable_transition) + len(transition2keep4net1)

        total_transition_num2 = total_trans_num - len(all_reachable_transition) + len(transition2keep4net2)

        overlap_idx = common_transition / min(total_transition_num1, total_transition_num2)

        if overlap_idx < min_overlap:
            min_overlap = overlap_idx
            transition2remove4net1 = all_reachable_transition.difference(transition2keep4net1)
            place2remove4net1 = all_reachable_place.difference(place2keep4net1)
            transition2remove4net2 = all_reachable_transition.difference(transition2keep4net2)
            place2remove4net2 = all_reachable_place.difference(place2keep4net2)

            transition2use = each_out_arc.target

    sub_net1 = PetriNet("The first sub Petri net")
    copied_net1 = __select_into(original_net, sub_net1, transition2remove4net1, place2remove4net1)
    sub_net2 = PetriNet("The second sub Petri net")
    copied_net2 = __select_into(original_net, sub_net2, transition2remove4net2, place2remove4net2)

    #   return a max decomposition: start place, merge place, transition and the least overlap value
    return copied_net1, copied_net2, min_overlap


def __select_into(source_net, target_net, transitions2remove, places2remove):
    t_map = {}
    p_map = {}
    for t in source_net.transitions:
        if t.name in transitions2remove:
            continue
        name = t.name
        label = t.label
        t_map[t] = PetriNet.Transition(name, label)
        if properties.TRACE_NET_TRANS_INDEX in t.properties:
            # 16/02/2021: copy the index property from the transition of the trace net
            t_map[t].properties[properties.TRACE_NET_TRANS_INDEX] = t.properties[properties.TRACE_NET_TRANS_INDEX]
        target_net.transitions.add(t_map[t])

    for p in source_net.places:
        if p.name in places2remove:
            continue
        name = p.name
        p_map[p] = PetriNet.Place(name)
        if properties.TRACE_NET_PLACE_INDEX in p.properties:
            # 16/02/2021: copy the index property from the place of the trace net
            p_map[p].properties[properties.TRACE_NET_PLACE_INDEX] = p.properties[properties.TRACE_NET_PLACE_INDEX]
        target_net.places.add(p_map[p])

    for t in source_net.transitions:
        if t.name in transitions2remove:
            continue
        for a in t.in_arcs:
            if a.source.name in places2remove:
                continue
            add_arc_from_to(p_map[a.source], t_map[t], target_net)
        for a in t.out_arcs:
            if a.target.name in places2remove:
                continue
            add_arc_from_to(t_map[t], p_map[a.target], target_net)

    return target_net


def __copy_into(source_net, target_net):
    t_map = {}
    p_map = {}
    for t in source_net.transitions:
        name = t.name
        label = t.label
        t_map[t] = PetriNet.Transition(name, label)
        if properties.TRACE_NET_TRANS_INDEX in t.properties:
            # 16/02/2021: copy the index property from the transition of the trace net
            t_map[t].properties[properties.TRACE_NET_TRANS_INDEX] = t.properties[properties.TRACE_NET_TRANS_INDEX]
        target_net.transitions.add(t_map[t])

    for p in source_net.places:
        name = p.name
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

    return target_net


def __find_all_reachable_transitions_dfs(start_place, target_place):
    visited = set()
    transitions_in_path = set()
    places_in_path = set()

    def bfs(node):
        if node in visited:
            return
        # add the node to the visited set
        visited.add(node)
        # if it is the target place, then we continue
        if node == target_place:
            return
        if isinstance(node, PetriNet.Transition):
            transitions_in_path.add(node.name)
        if isinstance(node, PetriNet.Place):
            places_in_path.add(node.name)
        for arc in node.out_arcs:
            bfs(arc.target)
        return False

    for out_arc in start_place.out_arcs:
        bfs(out_arc.target)
    return transitions_in_path, places_in_path


def find_transitions_dfs(start_node, target_place):
    visited = set()
    transitions_in_path = set()
    places_in_path = set()

    def bfs(node):
        if node in visited:
            return
        # add the node to the visited set
        visited.add(node)
        # if it is the target place, then we continue
        if node == target_place:
            return
        if isinstance(node, PetriNet.Transition):
            transitions_in_path.add(node.name)
        if isinstance(node, PetriNet.Place):
            places_in_path.add(node.name)
        for arc in node.out_arcs:
            bfs(arc.target)
        return False

    bfs(start_node)
    return transitions_in_path, places_in_path


def get_reduced_pair(net):
    '''
    get the valid decomposable choice place pairs
    :param net: the petri net
    :return: the set of valid decomposable choice place pairs
    '''

    start_places_set, merge_places_set = get_start_split_place(net)
    valid_decomposable_pair = {}
    for start_place in start_places_set:
        valid_decomposable_pair[start_place] = []
        for merge_place in merge_places_set:
            if is_valid_split_pair(start_place, merge_place, len(net.places)):
                valid_decomposable_pair[start_place].append(merge_place)

    transitive_merge_pair = {}
    for mp1 in merge_places_set:
        transitive_merge_pair[mp1] = []
        for mp2 in merge_places_set:
            if mp1 != mp2:
                if is_transitive_merge_pair(mp1, mp2, len(net.places)):
                    transitive_merge_pair[mp1].append(mp2)

    reduced_valid_pair = {}
    for k, v in valid_decomposable_pair.items():
        if len(v) == 0:
            continue
        max_val = 0
        max_place = v[0]
        for each_place in v:
            if len(transitive_merge_pair[each_place]) > max_val:
                max_place = each_place
                max_val = len(transitive_merge_pair[each_place])
        reduced_valid_pair[k] = max_place
    return reduced_valid_pair


def decompose_into_k_subnet(petri_net, target_num):
    sub_net_lst = [petri_net]

    while len(sub_net_lst) < target_num:
        target_net = get_net_with_maximum_transitions(sub_net_lst)
        sub_net_lst.remove(target_net)
        reduced_valid_pair = get_reduced_pair(target_net)
        net1, net2, min_overlap1 = None, None, 1

        for k, v in reduced_valid_pair.items():
            temp_net1, temp_net2, temp_min_overlap = extract_subnet_along_outgoing_edges(target_net, k, v)
            if temp_min_overlap < min_overlap1:
                min_overlap1 = temp_min_overlap
                net1, net2 = temp_net1, temp_net2
        sub_net_lst.append(net1)
        sub_net_lst.append(net2)

    return sub_net_lst


def get_net_with_maximum_transitions(sub_net_lst):
    max_trans = 0
    target_net = None
    for net in sub_net_lst:
        if len(net.transitions) > max_trans:
            max_trans = len(net.transitions)
            target_net = net
    return target_net


if __name__ == "__main__":
    net, im, fm = pnml_importer.apply("../model/prAm6_id.pnml")
    print("transition number: ", len(net.transitions), " place number: ", len(net.places),"\n")

    start_time = time.time()
    net_lst = decompose_into_k_subnet(net,4)

    idx = 1
    for each_net in net_lst:
        print("transition number: ", len(each_net.transitions), " place number: ", len(each_net.places))
        # gviz = visualizer.apply(net1)
        # visualizer.view(gviz)
        pnml_exporter.apply(each_net, im, "../model/prAm6_id_sub" + str(idx)+ ".pnml")
        idx += 1
    # pnml_exporter.apply(net2, im, "../model/hospital_id02_subnet2.pnml")

    print("time: ", time.time()-start_time)