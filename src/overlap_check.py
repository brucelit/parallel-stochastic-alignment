def get_transition_overlap(pn1_transition, pn2_transition):
    '''

    Parameters
    ----------
    pn1_transition: transitions in pn1
    pn2_transition: transitions in pn2

    Returns
    -------
    the minimum overlap in two petri nets
    '''
    overlap_count = 0
    for trans in pn1_transition:
        if trans in pn2_transition:
            overlap_count += 1

    overlap_percentage1 = overlap_count/len(pn1_transition)
    overlap_percentage2 = overlap_count/len(pn2_transition)
    return min(overlap_percentage1, overlap_percentage2)