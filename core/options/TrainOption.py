from __future__ import annotations
from ast import arg

from core.utils.processingUtil import HLAProcessor, PeptideGenerator

from typing import Dict, List, Literal, Union
import argparse


class TrainOption(object):
    """
    Argument for training

    Note: every variable here need to has type annotation so that we can parsed from argparse
    """
    dataset: Literal['MSI011320', 'ANTI051821Z', 'MSI011320_ANTI051821Z_COMBINE', 'PRETRAIN_3D', 'PRETRAIN_3D-HUMAN_PEPTIDE-TWO_HEAD'] = None
    root_dir: str = '/media/zen3515/Data/Zenthesis/datasets/'
    save_path: str = 'experiment1'
    experiment_name: str = 'experiment1_train'
    epoch: int = 1050
    batch_size_train: int = 256
    batch_size_test: int = 256
    optimizer: Literal['SGD'] = 'SGD'
    learning_rate: float = 1e-2
    loss: Literal['binary_crossentropy'] = 'binary_crossentropy'
    metrics: List[Literal['acc', 'AUC']] = ['acc', 'AUC']
    load_embedding_peptide: bool = False
    load_embedding_allele: bool = False
    embedding_peptide_path: str = 'resources/intermediate_netmhc2/peptide_central_embedding.npy'
    embedding_allele_path: str = 'resources/intermediate_netmhc2/allele_central_embedding.npy'
    # fit config
    train_use_multiprocessing: bool = False  # useful for training pretraining as it use generator
    train_num_workers: int = 1
    train_max_queue_size: int = 10  # should be 10*number of worker
    # callback
    checkpoint_monitor: str = 'val_auc'
    reduce_lr_monitor: str = 'val_acc'
    reduce_lr_patience: int = 11
    reduce_lr_min_delta: float = 0.0001
    reduce_lr_factor: float = 0.8
    early_stop_monitor: str = 'val_acc'
    early_stop_patience: int = 32
    weight_freezer_layers: list[str] = []  # ['peptide_embeddings', 'allele_embeddings']
    weight_freezer_epoch: int = 0
    # MSI011320 option
    kfold: int = 5
    run_kfold: List[int] = []
    gen_neg_ratio: float = 0.4
    # PRETRAIN_3D option
    central2context_path: str = "/media/zen3515/Data/Zenthesis/datasets/PRETRAIN_3D/dist_threshold_0p3/central2context.yaml"
    pair_map_counter_path: str = "/media/zen3515/Data/Zenthesis/datasets/PRETRAIN_3D/dist_threshold_0p3/pair_map_counter.yaml"
    # PRETRAIN_3D-HUMAN_PEPTIDE-TWO_HEAD option
    human_peptide_central2context_path: str = "/media/zen3515/Data/Zenthesis/datasets/PRETRAIN_HUMAN_PROTEIN/human-protein_word-size-3_window-size-3/central2context.yaml"
    human_peptide_pair_map_counter_path: str = "/media/zen3515/Data/Zenthesis/datasets/PRETRAIN_HUMAN_PROTEIN/human-protein_word-size-3_window-size-3/pair_map_counter.yaml"
    # shared with inference
    peptide_lenght_dist_path: str = 'resources/intermediate_netmhc2/peptide_lenght_dist.yaml'
    MODEL_TYPE: Literal['MHCSeqNet2', 'MHCSeqNet2_GRUPeptide', 'GloVeFastText', 'MultiHeadGloVeFastTextSplit', 'MultiHeadGloVeFastTextJointed'] = 'MHCSeqNet2'
    ALLELE_MAPPER_PATH: str = 'resources/allele_mapper'
    NEGATIVE_AMINO_PATH: str = 'resources/NEGATIVE_AMINO.txt'
    VOCAB_PATH: str = 'resources/VOCAB.txt'
    VOCAB_ALLELE_PATH: str = 'resources/VOCAB_ALLELE.txt'

    def __repr__(self) -> str:
        return (
            "TrainOption:\n"
            f"  dataset: {TrainOption.dataset}\n"
            f"  root_dir: {TrainOption.root_dir}\n"
            f"  save_path: {TrainOption.save_path}\n"
            f"  experiment_name: {TrainOption.experiment_name}\n"
            f"  epoch: {TrainOption.epoch}\n"
            f"  batch_size_train: {TrainOption.batch_size_train}\n"
            f"  batch_size_test: {TrainOption.batch_size_test}\n"
            f"  optimizer: {TrainOption.optimizer}\n"
            f"  learning_rate: {TrainOption.learning_rate}\n"
            f"  loss: {TrainOption.loss}\n"
            f"  metrics: {TrainOption.metrics}\n"
            f"  load_embedding_peptide: {TrainOption.load_embedding_peptide}\n"
            f"  load_embedding_allele: {TrainOption.load_embedding_allele}\n"
            f"  embedding_peptide_path: {TrainOption.embedding_peptide_path}\n"
            f"  embedding_allele_path: {TrainOption.embedding_allele_path}\n"
            f"  train_use_multiprocessing: {TrainOption.train_use_multiprocessing}\n"
            f"  train_num_workers: {TrainOption.train_num_workers}\n"
            f"  train_max_queue_size: {TrainOption.train_max_queue_size}\n"
            f"  checkpoint_monitor: {TrainOption.checkpoint_monitor}\n"
            f"  reduce_lr_monitor: {TrainOption.reduce_lr_monitor}\n"
            f"  reduce_lr_patience: {TrainOption.reduce_lr_patience}\n"
            f"  reduce_lr_min_delta: {TrainOption.reduce_lr_min_delta}\n"
            f"  reduce_lr_factor: {TrainOption.reduce_lr_factor}\n"
            f"  early_stop_monitor: {TrainOption.early_stop_monitor}\n"
            f"  early_stop_patience: {TrainOption.early_stop_patience}\n"
            f"  weight_freezer_layers: {TrainOption.weight_freezer_layers}\n"
            f"  weight_freezer_epoch: {TrainOption.weight_freezer_epoch}\n"
            f"  kfold: {TrainOption.kfold}\n"
            f"  run_kfold: {TrainOption.run_kfold}\n"
            f"  gen_neg_ratio: {TrainOption.gen_neg_ratio}\n"
            f"  central2context_path: {TrainOption.central2context_path}\n"
            f"  pair_map_counter_path: {TrainOption.pair_map_counter_path}\n"
            f"  human_peptide_central2context_path: {TrainOption.human_peptide_central2context_path}\n"
            f"  human_peptide_pair_map_counter_path: {TrainOption.human_peptide_pair_map_counter_path}\n"
            f"  peptide_lenght_dist_path: {TrainOption.peptide_lenght_dist_path}\n"
            f"  MODEL_TYPE: {TrainOption.MODEL_TYPE}\n"
            f"  ALLELE_MAPPER_PATH: {TrainOption.ALLELE_MAPPER_PATH}\n"
            f"  NEGATIVE_AMINO_PATH: {TrainOption.NEGATIVE_AMINO_PATH}\n"
            f"  VOCAB_PATH: {TrainOption.VOCAB_PATH}\n"
            f"  VOCAB_ALLELE_PATH: {TrainOption.VOCAB_ALLELE_PATH}\n"
        ).strip()

    @classmethod
    def parseArgs(cls, args: TrainOption) -> None:
        """
        args is from argparser.parse_args(),
        use to fill the class variable
        """
        # need to be here for main to set_memory_growth first
        import tensorflow as tf
        METRICS_DICT: Dict[str, Union[tf.keras.metrics.AUC, Literal['acc']]] = {
            'AUC': tf.keras.metrics.AUC(),
            'acc': 'acc'
        }
        if args.dataset == 'MSI011320' or args.dataset == 'MSI011320_ANTI051821Z_COMBINE' or args.dataset == "PRETRAIN_3D" or args.dataset == "PRETRAIN_3D-HUMAN_PEPTIDE-TWO_HEAD":
            HLAProcessor.load_from_disk(args.VOCAB_ALLELE_PATH, args.ALLELE_MAPPER_PATH)
            PeptideGenerator.load_from_disk(args.VOCAB_PATH, args.peptide_lenght_dist_path, args.NEGATIVE_AMINO_PATH)
        for prop in cls.__annotations__.keys():
            setattr(cls, prop, getattr(args, prop))
        if len(args.run_kfold) == 0:  # if not specify, run all
            cls.run_kfold = list(range(1, args.kfold+1))
            args.run_kfold = cls.run_kfold
        cls.metrics = [METRICS_DICT[metric] for metric in cls.metrics]
        args.metrics = cls.metrics

    # @classmethod
    # def getArgumentsString(cls, sep: typing.Literal[' ', '='] = '=') -> List[str]:
    #     return [f"--{propName}{sep}{propValue}" if propType != "bool" else f"--{propName}" for propName, propType in cls.__annotations__.items() if (propValue := getattr(cls, propName, None)) is not None]


trainParser = argparse.ArgumentParser(description='MHCSeqNet2 Trainer')

# Normal args
trainParser.add_argument('--dataset', required=True, choices=['MSI011320', 'ANTI051821Z', 'MSI011320_ANTI051821Z_COMBINE',
                         'PRETRAIN_3D', 'PRETRAIN_3D-HUMAN_PEPTIDE-TWO_HEAD'], help=f'dataset use for training')
trainParser.add_argument('--root_dir', default=TrainOption.root_dir, type=str, help=f'root folder containing both `MSI011320` and `ANTI051821Z`')
trainParser.add_argument('--save_path', default=TrainOption.save_path, type=str, help=f'folder to save the model weight')
trainParser.add_argument('--experiment_name', default=TrainOption.experiment_name, type=str, help=f'experiment name for share memory and folder to save')
trainParser.add_argument('--epoch', default=TrainOption.epoch, type=int, help='total number of training epoch')
trainParser.add_argument('--batch_size_train', default=TrainOption.batch_size_train, type=int, help='training batch size')
trainParser.add_argument('--batch_size_test', default=TrainOption.batch_size_test, type=int, help='testing batch size')
trainParser.add_argument('--optimizer', default=TrainOption.optimizer, type=str, choices=['SGD'], help=f'specify optimizer')
trainParser.add_argument('--learning_rate', default=TrainOption.learning_rate, type=float, help=f'specify learning_rate')
trainParser.add_argument('--loss', default=TrainOption.loss, type=str, choices=['binary_crossentropy'], help=f'specify loss')
trainParser.add_argument('--metrics', default=TrainOption.metrics, type=str, nargs='+', help=f'specify metrics')
trainParser.add_argument('--load_embedding_peptide', default=TrainOption.load_embedding_peptide, action='store_true', help=f'load pretrain embedding matrix of peptide')
trainParser.add_argument('--load_embedding_allele', default=TrainOption.load_embedding_allele, action='store_true', help=f'load pretrain embedding matrix of allele')
trainParser.add_argument('--embedding_peptide_path', default=TrainOption.embedding_peptide_path, type=str, help=f'path to embedding matrix.npy of peptide')
trainParser.add_argument('--embedding_allele_path', default=TrainOption.embedding_allele_path, type=str, help=f'path to embedding matrix.npy of allele')
trainParser.add_argument('--train_use_multiprocessing', default=TrainOption.train_use_multiprocessing, action='store_true', help=f'useful for training pretraining as it use generator')
trainParser.add_argument('--train_num_workers', default=TrainOption.train_num_workers, type=int, help='useful for training pretraining as it use generator')
trainParser.add_argument('--train_max_queue_size', default=TrainOption.train_max_queue_size, type=int, help='useful for training pretraining as it use generator')
trainParser.add_argument('--checkpoint_monitor', default=TrainOption.checkpoint_monitor, type=str, help='train callback setting')
trainParser.add_argument('--reduce_lr_monitor', default=TrainOption.reduce_lr_monitor, type=str, help='train callback setting')
trainParser.add_argument('--reduce_lr_patience', default=TrainOption.reduce_lr_patience, type=int, help='train callback setting')
trainParser.add_argument('--reduce_lr_min_delta', default=TrainOption.reduce_lr_min_delta, type=float, help='train callback setting')
trainParser.add_argument('--reduce_lr_factor', default=TrainOption.reduce_lr_factor, type=float, help='train callback setting')
trainParser.add_argument('--early_stop_monitor', default=TrainOption.early_stop_monitor, type=str, help='train callback setting')
trainParser.add_argument('--early_stop_patience', default=TrainOption.early_stop_patience, type=int, help='train callback setting')
trainParser.add_argument('--weight_freezer_layers', default=TrainOption.weight_freezer_layers, type=str, nargs='+', help=f'freezeing layer names')
trainParser.add_argument('--weight_freezer_epoch', default=TrainOption.weight_freezer_epoch, type=int, help='how long until we unfreeze it')
trainParser.add_argument('--kfold', default=TrainOption.kfold, type=int, help='When use --dataset=MSI011320\nSpecify number of cross validation fold')
trainParser.add_argument('--run_kfold', default=TrainOption.run_kfold, type=int, nargs='+',
                         help=f'the list of kfold to run. So that you can train all kfold at the same time in different process')
trainParser.add_argument('--gen_neg_ratio', default=TrainOption.gen_neg_ratio, type=float, help='When use --dataset=MSI011320\nSpecify ratio of negative class ganeration')
trainParser.add_argument('--central2context_path', default=TrainOption.central2context_path, type=str, help=f'path to central to context for pretrain 3D data')
trainParser.add_argument('--pair_map_counter_path', default=TrainOption.pair_map_counter_path, type=str, help=f'path to pair_map_counter for pretrain 3D data')
trainParser.add_argument('--human_peptide_central2context_path', default=TrainOption.human_peptide_central2context_path, type=str, help=f'path to human peptide central to context for pretrain data')
trainParser.add_argument('--human_peptide_pair_map_counter_path', default=TrainOption.human_peptide_pair_map_counter_path, type=str, help=f'path to human peptide pair_map_counter for pretrain data')
trainParser.add_argument('--peptide_lenght_dist_path', default=TrainOption.peptide_lenght_dist_path, type=str, help='path to prepared peptide length distribution')
trainParser.add_argument('--MODEL_TYPE', default=TrainOption.MODEL_TYPE, type=str, choices=['MHCSeqNet2',
                         'MHCSeqNet2_GRUPeptide', 'GloVeFastText', 'MultiHeadGloVeFastTextSplit', 'MultiHeadGloVeFastTextJointed'], help='specify model to use')
trainParser.add_argument('--ALLELE_MAPPER_PATH', default=TrainOption.ALLELE_MAPPER_PATH, type=str,
                         help='path to the folder that contain yaml file needed for the tool.\nYou can use this to add a new allele, please visit readme for more')
trainParser.add_argument('--NEGATIVE_AMINO_PATH', default=TrainOption.NEGATIVE_AMINO_PATH, type=str, help='path to prepared amino acide used for generating negative data')
trainParser.add_argument('--VOCAB_PATH', default=TrainOption.VOCAB_PATH, type=str, help='path to prepared vocab of the model')
trainParser.add_argument('--VOCAB_ALLELE_PATH', default=TrainOption.VOCAB_ALLELE_PATH, type=str, help='path to prepared vocab of alleles of the model')
# trainParser.add_argument('--GPU_ID', default=TrainOption.GPU_ID, type=int, help='default GPU, you can specify a GPU to be used by given a number i.e, `--GPU_ID 0`')
# trainParser.add_argument('--USE_ENSEMBLE', default=TrainOption.USE_ENSEMBLE, action='store_true', help='blablabla')
