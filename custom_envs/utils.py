def crop_or_pad_to_maxlen(it, maxlen: int, pad_item=0):
    return (it + [pad_item] * (maxlen - len(it)))[:maxlen]
