from tensorflow.keras.optimizers import SGD
from core.options import TrainOption

import typing


def build_optimizer(args: TrainOption):
    if args.optimizer == 'SGD':
        return SGD(learning_rate=args.learning_rate)
    else:
        raise AssertionError(f'Unsupported optimizer {args.optimizer}')
