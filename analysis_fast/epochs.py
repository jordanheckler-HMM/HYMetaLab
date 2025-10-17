def slice_by_epoch(df):
    groups = {}
    for e, g in df.groupby("epoch"):
        groups[int(e)] = g.reset_index(drop=True)
    return groups
