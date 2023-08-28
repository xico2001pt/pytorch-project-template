from torch.optim import SGD, Adam


class SGDOptimizer(SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AdamOptimizer(Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
