from .losses import CrossEntropyLoss
from .metrics import AccuracyMetric
from .optimizers import SGDOptimizer, AdamOptimizer
from .schedulers import ExponentialLRScheduler
from .stop_conditions import StopPatience

classes = {
    "losses": [CrossEntropyLoss],  # Add the loss classes here
    "metrics": [AccuracyMetric],  # Add the metric classes here
    "optimizers": [SGDOptimizer, AdamOptimizer],  # Add the optimizer classes here
    "schedulers": [ExponentialLRScheduler],  # Add the scheduler classes here
    "stop_conditions": [StopPatience],  # Add the stop condition classes here
}
