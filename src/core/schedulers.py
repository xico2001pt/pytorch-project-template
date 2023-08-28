from torch.optim.lr_scheduler import ExponentialLR


class ExponentialLRScheduler(ExponentialLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
