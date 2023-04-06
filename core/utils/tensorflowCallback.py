from typing import Any, Dict, List
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping

import numpy as np
import logging
# import tensorflow as tf
import tracemalloc


class EpochMemoryDebuggerCallback(Callback):

    def __init__(self):
        super().__init__()
        self.snapshots: Dict[str, tracemalloc.Snapshot] = {}

    def on_train_begin(self, logs: Dict[str, Any] = None):
        # tracemalloc.start()
        self.snapshots['base'] = tracemalloc.take_snapshot()

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        self.snapshots[f'begin-{epoch}'] = tracemalloc.take_snapshot()

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        self.snapshots[f'end-{epoch}'] = tracemalloc.take_snapshot()
        top_stats = self.snapshots[f'end-{epoch}'].compare_to(self.snapshots[f'begin-{epoch}'], 'lineno')

        print("[ Top 20 differences ]")
        logging.debug("[ Top 20 differences ]")
        for stat in top_stats[:20]:
            print(stat)
            logging.debug(stat)


class EarlyStoppingIfMetricMax(EarlyStopping):

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False):
        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience or ('acc' in self.monitor and abs(current - 1) <= 1e-4):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)


class WeightFreezerCallback(Callback):

    def __init__(self, freezing_layers: List[str] = [], freeze_epoch: int = 2):
        super().__init__()
        self.model: Model
        self.freezing_layers = freezing_layers
        self.freeze_epoch = freeze_epoch

    def on_train_begin(self, logs=None):
        for freezeing_layer in self.freezing_layers:
            self.model.get_layer(freezeing_layer).trainable = False

    def on_train_end(self, logs=None):
        # reset it back
        for freezeing_layer in self.freezing_layers:
            self.model.get_layer(freezeing_layer).trainable = True

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.freeze_epoch:
            for frozen_layer in self.freezing_layers:
                self.model.get_layer(frozen_layer).trainable = True
