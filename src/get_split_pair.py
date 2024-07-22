


def get_start_split_place(petri_net):
    '''
    Parameters
    ----------
    petri_net

    Returns
    -------
    return the set of places with multiple outgoing/incoming edges
    '''

    start_places_set = set()
    merge_places_set = set()
    for place in petri_net.places:
#         if the place has multiple outgoing edges
        if place.out_arcs > 1:
            start_places_set.add(place)
        if place.in_arcs > 1:
            merge_places_set.add(place)
    return start_places_set, merge_places_set


def is_valid_split_pair(start_p, merge_p):
    is_in_loop = False

    to_visited_place_set = [start_p]
    visited_place_set = set()
    # conduct breadth first search from the start place start_p
    while len(to_visited_place_set)>0:
        place_to_visit =

    # if the place is already in the visited_place_set, continue


    return False
