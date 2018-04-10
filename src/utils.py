def split_list(l, total, i):
    persplit = len(l) // total
    offset = len(l) % total
    if i < offset:
        return l[persplit * i + i:persplit * (i + 1) + i + 1]
    else:
        return l[persplit * i + offset:persplit * (i + 1) + offset]
