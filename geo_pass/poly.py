def are_equal(p1, p2):
    return p1.lat == p2.lat and p1.lon == p2.lon


def ways2poly(ways):
    """
    Given an iterable of `ways`, combined them into one or more polygons.

    Args:
        ways: iterable of `overpy.Way` that form the desired polygon:w

    Return:
        polys, incomplete: `polys` is a list of list of  (long, lat) coords describing
            valid (i.e. closed) polygons; `incomplete` is a list of list of (long, lat)
            coords describing "incomplete polygons" (i.e. LineString)
    """
    w = set(ways)
    polys = []
    incomplete = []
    current = None
    while True:
        if not current:
            if len(w) > 0:
                current = w.pop().geometry
            else:
                break
        if current[0] == current[-1]:
            polys.append(current)
            current = None
            continue
        else:
            if len(w) < 1:
                incomplete.append(current)
                break
            to_remove = set()
            for n in w:
                if are_equal(n.geometry[0], current[-1]):
                    current += n.geometry
                elif are_equal(n.geometry[0], current[0]):
                    current = list(reversed(n.geometry)) + current
                elif are_equal(n.geometry[-1], current[0]):
                    current = n.geometry + current
                elif are_equal(n.geometry[-1], current[-1]):
                    current += list(reversed(n.geometry))
                else:
                    continue
                to_remove.add(n)
            if len(to_remove) == 0:
                incomplete.append(current)
                current = None
                continue
            w -= to_remove

    return polys, incomplete
