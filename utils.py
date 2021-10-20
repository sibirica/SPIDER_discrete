def save(filename, *args):
    with open(filename, 'wb') as f:
        for arr in args:
            np.save(f, arr)
            
def load(filename, nload):
    to_load = []
    with open(filename, 'rb') as f:
        for i in range(nload):
            to_load.append(np.load(f))
    return tuple(to_load)
