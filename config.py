hp_dict = dict(
    # this can be any network supported by efficientnet-pytorch-3d  
    arch = 'efficientnet-b0', 

    # optimizer settings
    use_sgd = False,
    lr = 0.001,
    momentum = 0.9,
)

class Config(object):
    def __init__(self, hp_dict):
        object.__setattr__(self, "_params", dict())
        for key in hp_dict:
            self[key] = hp_dict[key]

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, val):
        self._params[key] = val

    def __getattr__(self, key):
        return self._params[key]

    def get(self):
        return self._params
