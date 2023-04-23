"""
This is the main file use for spawn another training processes
"""
from math import ceil
from typing import List
from core.datasets import DATASET_TO_CLASS_FACTORY, build_pipeline
from core.models import build_model
from core.optimizer import build_optimizer
from core.options import TrainOption, trainParser
from core.utils import plotTraing
from multiprocessing import shared_memory
# from tensorflow.keras.utils import GeneratorEnqueuer
from tensorflow.keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, LambdaCallback, Callback
from core.utils.tensorflowCallback import EarlyStoppingIfMetricMax, WeightFreezerCallback

import os
import numpy as np
import logging
import tensorflow as tf

# import tracemalloc


def setup_logging(save_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(save_path),
            logging.StreamHandler()
        ]
    )


if __name__ == '__main__':
    # tracemalloc.start()
    # START_MEMORY_SNAPSHOT = tracemalloc.take_snapshot()

    # Tensorflow memory stuffs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    args: TrainOption = trainParser.parse_args()

    TrainOption.parseArgs(args)

    if TrainOption.dataset == 'MSI011320' or TrainOption.dataset == 'MSI011320_ANTI051821Z_COMBINE':
        dataset_cls = DATASET_TO_CLASS_FACTORY[TrainOption.dataset]
        for kfold in TrainOption.run_kfold:
            dataset_size = shared_memory.ShareableList(name=f'{TrainOption.dataset}_{TrainOption.experiment_name}_{kfold}_dataset_size', sequence=[500000]*3)
            save_folder = os.path.join(TrainOption.save_path, TrainOption.experiment_name, f"{TrainOption.experiment_name}_{kfold}-{TrainOption.kfold}")
            log_path = os.path.join(save_folder, 'train.log')
            tensorboard_path = os.path.join(save_folder, 'tensorboard_log')
            model_name = f'{TrainOption.MODEL_TYPE}_{kfold}_{TrainOption.kfold}'
            weight_path = os.path.join(save_folder, f'{model_name}.h5')
            prediction_test_path = os.path.join(save_folder, f'pred_test.csv')
            os.makedirs(save_folder, exist_ok=True)
            os.makedirs(tensorboard_path, exist_ok=True)
            setup_logging(log_path)
            logging.info(f'k-fold: {kfold}/{TrainOption.kfold}\nweight will be saved at {os.path.abspath(weight_path)}')
            logging.info(f'{TrainOption()}')
            model = build_model(args)
            optim = build_optimizer(args)
            model.compile(optimizer=optim,
                          loss=TrainOption.loss,
                          metrics=TrainOption.metrics)
            trainPipeline, evalPipeline, testPipeline = build_pipeline(args, kfold=kfold)
            if TrainOption.epoch > 0:
                # Else we could just load weight and predict
                # callbacks
                callbacks_list: List[Callback] = []
                # Empirical result shown that monitor='val_acc' leads to better val auc and acc, so reduce_lr and early_stop used val_acc instead. we just save the best auc instead
                checkpoint = ModelCheckpoint(weight_path, monitor=TrainOption.checkpoint_monitor, verbose=1, save_weights_only=True, save_best_only=True, mode='max')
                reduce_lr = ReduceLROnPlateau(monitor=TrainOption.reduce_lr_monitor, patience=TrainOption.reduce_lr_patience, factor=TrainOption.reduce_lr_factor, min_lr=5e-4, verbose=1)
                early_stop = EarlyStopping(monitor=TrainOption.early_stop_monitor, min_delta=0.0001, patience=TrainOption.early_stop_patience, verbose=1, mode='max')
                callbacks_list.extend([checkpoint, reduce_lr, early_stop])
                if len(TrainOption.weight_freezer_layers) > 0:
                    weight_freezer = WeightFreezerCallback(freezing_layers=TrainOption.weight_freezer_layers, freeze_epoch=TrainOption.weight_freezer_epoch)
                    callbacks_list.append(weight_freezer)
                tensor_board = TensorBoard(log_dir=tensorboard_path)
                log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: logging.info(
                    f"[Epoch: {epoch}] lr={logs['lr']}, loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}, acc={logs['acc']:.4f}, val_acc={logs['val_acc']:.4f}, auc={logs['auc']:.4f}, val_auc={logs['val_auc']:.4f}"))
                # memory_debugger_callback = EpochMemoryDebuggerCallback()
                callbacks_list.extend([tensor_board, log_callback])
                dataset_size_train, dataset_size_eval, dataset_size_test = dataset_size
                logging.info(f'dataset_size_train {dataset_size_train}, dataset_size_eval {dataset_size_eval}, dataset_size_test {dataset_size_test}')
                callback_hist: History = model.fit(x=trainPipeline,
                                                steps_per_epoch=int(dataset_size_train / (TrainOption.batch_size_train - int(TrainOption.batch_size_train * TrainOption.gen_neg_ratio))),
                                                verbose=1,
                                                validation_data=evalPipeline,
                                                validation_steps=int(dataset_size_eval / TrainOption.batch_size_test),
                                                epochs=TrainOption.epoch,
                                                callbacks=callbacks_list)
                plotTraing(callback_hist, k_fold_num=kfold, save_path=save_folder)
                logging.info((f'Training best epoch at {callback_hist.epoch[np.argmax(callback_hist.history[TrainOption.early_stop_monitor])]}'
                            f', with max {TrainOption.early_stop_monitor} = {np.max(callback_hist.history[TrainOption.early_stop_monitor])}'))
            # Load best weight before making the prediction
            model.load_weights(weight_path)
            prediction = model.predict(testPipeline, steps=int(dataset_size_test / TrainOption.batch_size_test), verbose=1)
            dataset_cls.save_result(args, kfold, prediction_test_path, prediction)
            dataset_size.shm.close()
            dataset_size.shm.unlink()

            # STOP_MEMORY_SNAPSHOT = tracemalloc.take_snapshot()

            # top_stats = STOP_MEMORY_SNAPSHOT.compare_to(START_MEMORY_SNAPSHOT, 'filename')
            # [print(stat) for stat in top_stats[:20]]
            # breakpoint()
    elif TrainOption.dataset == 'ANTI051821Z':
        pass  # TODO
    elif TrainOption.dataset in ['PRETRAIN_3D', 'PRETRAIN_3D-HUMAN_PEPTIDE-TWO_HEAD']:
        dataset_size = shared_memory.ShareableList(name=f'{TrainOption.dataset}_{TrainOption.experiment_name}_dataset_size', sequence=[1018112, None, None])
        save_folder = os.path.join(TrainOption.save_path, TrainOption.experiment_name)
        log_path = os.path.join(save_folder, 'train.log')
        tensorboard_path = os.path.join(save_folder, 'tensorboard_log')
        model_name = f'{TrainOption.MODEL_TYPE}'
        weight_path = os.path.join(save_folder, f'{model_name}.h5')
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(tensorboard_path, exist_ok=True)
        setup_logging(log_path)
        logging.info(f'weight will be saved at {os.path.abspath(weight_path)}')
        logging.info(f'{TrainOption()}')
        model = build_model(args)
        optim = build_optimizer(args)
        if TrainOption.dataset == 'PRETRAIN_3D-HUMAN_PEPTIDE-TWO_HEAD':
            # loss per output can be apply with loss_weights I'll have to, or I can even wirte my own loss
            # TODO: when using TWO_HEAD, the monitor must be changed to `output_3d_acc` or `output_human-protien_acc` (loss,output_3d_loss,output_human-protien_loss,output_3d_acc,output_3d_auc,output_human-protien_acc,output_human-protien_auc,lr)
            model.compile(optimizer=optim,
                          loss=TrainOption.loss,
                          metrics=TrainOption.metrics,
                          loss_weights=[0.5, 0.5])
            log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: logging.info((f"[Epoch: {epoch}] lr={logs['lr']}, loss={logs['loss']:.4f}, "
                                                                                         f"output_3d_loss={logs['output_3d_loss']:.4f}, output_human-protien_loss={logs['output_human-protien_loss']:.4f}, "
                                                                                         f"output_3d_acc={logs['output_3d_acc']:.4f}, output_human-protien_acc={logs['output_human-protien_acc']:.4f}, "
                                                                                         f"output_3d_auc={logs['output_3d_auc']:.4f}, output_human-protien_auc={logs['output_human-protien_auc']:.4f}")))
        else:
            model.compile(optimizer=optim,
                          loss=TrainOption.loss,
                          metrics=TrainOption.metrics)
            log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: logging.info(f"[Epoch: {epoch}] lr={logs['lr']}, loss={logs['loss']:.4f}, acc={logs['acc']:.4f}, auc={logs['auc']:.4f}"))
        trainPipeline, _evalPipeline, _testPipeline = build_pipeline(args)
        # callbacks
        checkpoint = ModelCheckpoint(weight_path, monitor=TrainOption.checkpoint_monitor, verbose=1, save_weights_only=True, save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor=TrainOption.reduce_lr_monitor, patience=TrainOption.reduce_lr_patience,
                                      factor=TrainOption.reduce_lr_factor, min_delta=TrainOption.reduce_lr_min_delta, min_lr=5e-4, verbose=1)
        early_stop = EarlyStoppingIfMetricMax(monitor=TrainOption.early_stop_monitor, min_delta=0.0001, patience=TrainOption.early_stop_patience, verbose=1, mode='max')
        tensor_board = TensorBoard(log_dir=tensorboard_path)
        # memory_debugger_callback = EpochMemoryDebuggerCallback()
        dataset_size_train, dataset_size_eval, dataset_size_test = dataset_size
        logging.info(f'dataset_size_train {dataset_size_train}, dataset_size_eval {dataset_size_eval}, dataset_size_test {dataset_size_test}')
        if TrainOption.train_use_multiprocessing:
            logging.info(f'Multiprocessing is being used, this may cause memory leak, but if you have the ram for the leak, it is substantially faster')
        #     enq = GeneratorEnqueuer(trainPipeline, use_multiprocessing=True)
        #     enq.start(workers=TrainOption.train_num_workers, max_queue_size=TrainOption.train_max_queue_size)
        #     trainPipeline = enq.get()
        callback_hist: History = model.fit(x=trainPipeline,
                                           steps_per_epoch=ceil(dataset_size_train + 1 / TrainOption.batch_size_train),
                                           verbose=1,
                                           # validation_data=evalPipeline,
                                           # validation_steps=int(dataset_size_eval / TrainOption.batch_size_test),
                                           epochs=TrainOption.epoch,
                                           use_multiprocessing=TrainOption.train_use_multiprocessing,
                                           workers=TrainOption.train_num_workers,
                                           max_queue_size=TrainOption.train_max_queue_size,
                                           callbacks=[checkpoint, reduce_lr, early_stop, tensor_board, log_callback])  # memory_debugger_callback
        plotTraing(callback_hist, k_fold_num=0, save_path=save_folder)
        logging.info((f'Training best epoch at {callback_hist.epoch[np.argmax(callback_hist.history[TrainOption.early_stop_monitor])]}'
                      f', with max {TrainOption.early_stop_monitor} = {np.max(callback_hist.history[TrainOption.early_stop_monitor])}'))
        # prediction = model.predict(testPipeline, steps=int(dataset_size_test / TrainOption.batch_size_test), verbose=1)
        # Load best weight before making the prediction
        model.load_weights(weight_path)
        # save the result to a npy file
        central_embeddings_matrix: np.ndarray = np.copy(model.get_layer('central_embeddings').get_weights()[0])  # we use central embedding, but why?
        context_embeddings_matrix: np.ndarray = np.copy(model.get_layer('context_embeddings').get_weights()[0])
        central_embeddings_matrix_path = os.path.join(save_folder, 'central_embeddings_matrix.npy')
        context_embeddings_matrix_path = os.path.join(save_folder, 'context_embeddings_matrix.npy')
        np.save(central_embeddings_matrix_path, central_embeddings_matrix)
        np.save(context_embeddings_matrix_path, context_embeddings_matrix)
        logging.info(f"saved embedding at:\n{central_embeddings_matrix_path}\n{context_embeddings_matrix_path}")

        dataset_size.shm.close()
        dataset_size.shm.unlink()
