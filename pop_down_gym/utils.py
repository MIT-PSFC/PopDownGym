def remap_range(value, old_range, new_range):
    old_min = old_range[0]
    old_max = old_range[-1]
    new_min = new_range[0]
    new_max = new_range[-1]
    old_diff = old_max - old_min
    new_diff = new_max - new_min
    return (((value - old_min) * new_diff) / old_diff) + new_min