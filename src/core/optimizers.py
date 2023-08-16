from torch.optim import SGD, Adam


class SGD(SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Adam(Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
