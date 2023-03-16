def is_within_bb(bb, x, y):
    return bb[0] < x < bb[0] + bb[2] and bb[1] < y < bb[1] + bb[3]
