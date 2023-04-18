from __future__ import annotations

from core.options.TrainOption import TrainOption
from core.utils.processingUtil import HLAProcessor, PeptideGenerator
from multiprocessing import shared_memory

import os
import typing
import logging
# import tracemalloc
import tensorflow as tf
import pandas as pd
import numpy as np


class CSVDataset(tf.data.Dataset):
    """
    A dataset from base file `HLA_classI_MS_dataset_011320.tsv`
    # Parameters:
      - kfold: int - tell which fold to load `train_{k}.csv` and `test_{k}.csv`
    """
    home_dir: str = None  # Each child must override this
    root_dir = '/media/zen3515/Data/Zenthesis/datasets/'
    include_unknown_allele = False
    DEFAULT_ALLELE_COL = 'Allele'
    DEFAULT_PEPTIDE_COL = 'Peptide'
    # MEMORY_SNAPSHOTS: list[tracemalloc.Snapshot] = []

    @classmethod
    def _generator(cls,
                   kfold: int,
                   csv_path: bytes,
                   experiment_name: str,
                   phase: typing.Literal['train', 'eval', 'test'],
                   batch_size: int,
                   remainder_method: typing.Literal['repeat', 'sample'] = 'repeat',
                   is_map_allele2index: bool = True,
                   is_map_peptide2index: bool = True,
                   random_state: typing.Union[int, np.random.BitGenerator, np.random.RandomState] = -1,
                   will_log: bool = False):
        try:
            phase = phase.decode('UTF-8')
            remainder_method = remainder_method.decode('UTF-8')
            experiment_name = experiment_name.decode('UTF-8')
        except (UnicodeDecodeError, AttributeError):
            pass
        csv_path = csv_path.decode("utf-8")
        phase_index = ['train', 'eval', 'test'].index(phase)
        ALLELE_COL = cls.DEFAULT_ALLELE_COL

        def log(msg: str):
            if will_log:
                logging.info(msg)

        # Opening the file
        data_df: pd.DataFrame = pd.read_csv(csv_path, index_col=0, dtype={"HLA Allele": str, "Report Status": str, "IEDB Status": str, "Is Outlier": str,
                                            "has_binding_test": 'boolean', "isin_MSI011320": 'boolean', "is_outlier": 'boolean'})  # chunksize=num_positive # use chunk or batch() which one is better
        if not cls.include_unknown_allele:
            if 'isKnownAllele' not in data_df.columns:
                data_df.loc[:, 'isKnownAllele'] = data_df[ALLELE_COL].isin(HLAProcessor.mapper.keys())
            log(f"[{cls.__name__}] [{phase}] Removed unknown allele, dropeed {sum(data_df['isKnownAllele'] == False)}/{len(data_df)} rows")
            data_df = data_df.loc[data_df['isKnownAllele']]
        dataset_size = shared_memory.ShareableList(name=f'{cls.__name__}_{experiment_name}_{kfold}_dataset_size')
        dataset_size[phase_index] = len(data_df)

        # data_df = data_df.reset_index(drop=True)
        if is_map_allele2index:
            data_df.loc[:, 'Allele_index'] = data_df[ALLELE_COL].apply(lambda x: HLAProcessor.mapper2index[x])  # .astype(np.int32)
            ALLELE_COL = 'Allele_index'

        PEPTIDE_COL = cls.DEFAULT_PEPTIDE_COL
        if is_map_peptide2index:
            data_df.loc[:, 'Peptide_Mapped'] = data_df[PEPTIDE_COL].apply(lambda peptide: np.array(
                [PeptideGenerator.padding_index_value]*(PeptideGenerator.max_peptide_length - len(peptide)) + [PeptideGenerator.PEPTIDE_LEN_ONE_MAPPER_DICT[amino] for amino in peptide]))
            PEPTIDE_COL = 'Peptide_Mapped'

        if random_state != -1:
            data_df = data_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        if (remainder_count := (len(data_df) % batch_size)) != 0:
            log(f"[{cls.__name__}] [{phase}] Data doesn't fit in the last batch ({remainder_count} remaining), proceed by {remainder_method}")
            if remainder_method in ['repeat', b'repeat']:
                data_df = data_df.iloc[list(range(len(data_df))) + [-1]*(batch_size - remainder_count)]
            elif remainder_method in ['sample', b'sample']:
                neg_df = data_df.sample(n=(batch_size - remainder_count), random_state=random_state)
                # data_df = data_df.append(neg_df)
                data_df = pd.concat([data_df, neg_df])
            else:
                raise AssertionError(f"Unknow remainder_method: {remainder_method}")
        dataset_size[phase_index] = len(data_df)
        log(f"[{cls.__name__}] [{phase}] has total aligned number of data {len(data_df)} rows")

        if 'isGenerated' not in data_df.columns:
            assert 'inference' in experiment_name, "only during inference that we doesn't generate"
            data_df.loc[:, 'isGenerated'] = False

        # FIXME: This is where I tricked myself, isGenerated should be 1 not 0
        for row_allele, row_peptide, row_is_gen in zip(data_df[ALLELE_COL], data_df[PEPTIDE_COL], data_df['isGenerated'].apply(lambda isGenerated: 0 if isGenerated else 1)):
            yield row_allele, row_peptide, row_is_gen

        dataset_size.shm.close()

    @classmethod
    def initialize_dry_run(cls,
                           kfold: int,
                           experiment_name: str,
                           phase: typing.Literal['train', 'eval', 'test'],
                           batch_size: int,
                           remainder_method: typing.Literal['repeat', 'sample'] = 'repeat',
                           root_dir: str = '/media/zen3515/Data/Zenthesis/datasets/',
                           is_map_allele2index: bool = True,
                           is_map_peptide2index: bool = True,
                           random_state: typing.Union[int, np.random.BitGenerator, np.random.RandomState] = -1,
                           will_log: bool = True):
        """ 
        This function solely exist for two reasons  
        - Test that the pipeline work before running the whole system
        - Initialize dataset_size
        """
        csv_path = cls.get_csv_path(kfold, phase, root_dir).encode('utf-8')
        next(cls._generator(kfold=kfold, csv_path=csv_path, experiment_name=experiment_name, phase=phase, batch_size=batch_size,
             remainder_method=remainder_method, is_map_allele2index=is_map_allele2index, is_map_peptide2index=is_map_peptide2index, random_state=random_state, will_log=will_log))

    @classmethod
    def __new__(cls,
                kfold: int,
                experiment_name: str,
                phase: typing.Literal['train', 'eval', 'test'],
                batch_size: int,
                remainder_method: typing.Literal['repeat', 'sample'] = 'repeat',
                root_dir: str = '/media/zen3515/Data/Zenthesis/datasets/',
                is_map_allele2index: bool = True,
                is_map_peptide2index: bool = True,
                random_state: typing.Union[int, np.random.BitGenerator, np.random.RandomState] = -1,
                will_log: bool = False):
        # print(f'__new__ got the following args {(cls, kfold, experiment_name, phase, batch_size, remainder_method, root_dir, is_map_allele2index, is_map_peptide2index, random_state, will_log,)}')
        csv_path = cls.get_csv_path(kfold, phase, root_dir)
        return tf.data.Dataset.from_generator(
            cls._generator,
            # output_types=(tf.int32 if is_map_allele2index else tf.string, tf.string, tf.int8),
            output_signature=(tf.TensorSpec(shape=(), dtype=tf.int32 if is_map_allele2index else tf.string),
                              tf.TensorSpec(shape=(PeptideGenerator.max_peptide_length,) if is_map_peptide2index else (),
                                            dtype=tf.int32 if is_map_peptide2index else tf.string),
                              tf.TensorSpec(shape=(), dtype=tf.int8)),
            args=(kfold, csv_path, experiment_name, phase, batch_size, remainder_method, is_map_allele2index, is_map_peptide2index, random_state, will_log)
        )

    @classmethod
    def get_csv_path(cls, kfold: int, phase: typing.Literal['train', 'eval', 'test'], root_dir: typing.Optional[str] = None):
        raise NotImplementedError('Subclass need to override this function')
        # return os.path.join(root_dir or cls.root_dir, cls.home_dir, f"HLA_classI_MS_dataset_011320_antigen_information_051821_for_zen_processed_kf-{kfold}_{phase}.csv")

    @classmethod
    def save_result(cls, args: TrainOption, kfold: int, save_path: str, prediction: np.ndarray):
        csv_path = cls.get_csv_path(kfold, 'test', args.root_dir).encode('utf-8')
        raw_save_data = [(*row, pred) for pred, row in zip(np.squeeze(prediction), cls._generator(kfold, csv_path, args.experiment_name, 'test',
                                                                                                  args.batch_size_test, remainder_method='repeat',
                                                                                                  is_map_allele2index=False, is_map_peptide2index=False, random_state=-1, will_log=False))]
        last_row = raw_save_data[-1]
        last_index = len(raw_save_data) - 1
        for i in range(2, args.batch_size_test):
            if last_row != raw_save_data[-i]:
                last_index = len(raw_save_data) - i + 1
                break
        # trim it
        raw_save_data = raw_save_data[:last_index + 1]
        save_df = pd.DataFrame(raw_save_data, columns=['Allele', 'Peptide', 'isGenerated', 'Prediction']) # FIXME: This is where I tricked myself, the data from _generator is `isBind` not `isGenerated`
        save_df.to_csv(save_path)
