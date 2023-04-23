from core.datasets.csv_datasets import CSVDataset
from core.utils.processingUtil import HLAProcessor, PeptideGenerator
from core.utils.yamlHelper import YamlHelper
from core.options import TrainOption, MHCToolOption
from core.datasets.msi011320 import MSI011320
from core.datasets.msi011320_anti051821z_combine import MSI011320_ANTI051821Z_COMBINE

import os
import numpy as np
import random
import typing
import pandas as pd
import logging
import tensorflow as tf

from functools import partial
from multiprocessing import shared_memory
from math import ceil

DatasetType = typing.Union[MSI011320, MSI011320_ANTI051821Z_COMBINE, tf.data.Dataset]

DATASET_TO_CLASS_FACTORY: typing.Dict[str, DatasetType] = {
    'MSI011320': MSI011320,
    'MSI011320_ANTI051821Z_COMBINE': MSI011320_ANTI051821Z_COMBINE
}

# def allele2seq_stage(allele: tf.Tensor, peptide: tf.Tensor, isGenerated: tf.Tensor):
#     return tf.py_function(HLAProcessor.map_hla2peptide_eager, inp=[allele], Tout=tf.string), peptide, isGenerated


def fill_nagative_stage(positive_size: int, negative_size: int, allele: tf.Tensor, peptide: tf.Tensor, isGenerated: tf.Tensor):
    generated_peptide = PeptideGenerator.generate_peptide(negative_size)
    generated_allele_batch_idx: tf.Tensor = tf.random.uniform(shape=(negative_size,), minval=1, maxval=positive_size, dtype=tf.int32)
    generated_allele_idx = tf.gather(allele, generated_allele_batch_idx)
    peptide = tf.concat([peptide, generated_peptide], axis=0)
    allele = tf.concat([allele, generated_allele_idx], axis=0)
    isGenerated = tf.concat([isGenerated, tf.zeros(shape=(negative_size,), dtype=tf.int8)], axis=0)
    return allele, peptide, isGenerated


# def peptide_to_int32_stage(positive_size: int, allele: tf.Tensor, peptide: tf.Tensor, isGenerated: tf.Tensor):
#     mapped_peptide = PeptideGenerator.map_peptide2int_graph(peptide, positive_size)
#     return allele, mapped_peptide, isGenerated

# def peptide_to_int32_stage(allele: tf.Tensor, peptide: tf.Tensor, isGenerated: tf.Tensor):
#     """
#     The bottle neck stage (Python GIL)
#     I've moved it to the dataset side
#     """
#     mapped_peptide = tf.py_function(PeptideGenerator.map_peptide2int, inp=[peptide], Tout=tf.int32)
#     return allele, mapped_peptide, isGenerated


def allele_to_int32_stage(allele: tf.Tensor, peptide: tf.Tensor, isGenerated: tf.Tensor):
    mapped_allele = HLAProcessor.map_allele_index2int(allele)
    return mapped_allele, peptide, isGenerated


def index_to_subword_stage(allele: tf.Tensor, peptide: tf.Tensor, isGenerated: tf.Tensor):
    allele_len_three = tf.stack([allele[:, ::3], allele[:, 1::3], allele[:, 2::3]], axis=-1)
    allele_len_two = tf.stack([allele_len_three[:, :, 0:2], allele_len_three[:, :, 1:3]], axis=-1)
    allele_len_one = tf.stack([allele[:, ::3, tf.newaxis], allele[:, 1::3, tf.newaxis], allele[:, 2::3, tf.newaxis]], axis=-1)

    mapped_allele_len_one = tf.squeeze(tf.gather(HLAProcessor.len_one_subword_allele_mapper, allele_len_one, axis=-1))
    mapped_allele_len_two = tf.gather_nd(HLAProcessor.len_two_subword_allele_mapper, allele_len_two)
    mapped_allele_len_three = tf.gather_nd(HLAProcessor.len_three_subword_allele_mapper, allele_len_three)[:, :, tf.newaxis]
    mapped_allele = tf.concat([mapped_allele_len_one, mapped_allele_len_two, mapped_allele_len_three], axis=-1)

    peptide_len_three = tf.stack([peptide[:, ::3], peptide[:, 1::3], peptide[:, 2::3]], axis=-1)
    peptide_len_two = tf.stack([peptide_len_three[:, :, 0:2], peptide_len_three[:, :, 1:3]], axis=-1)
    peptide_len_one = tf.stack([peptide[:, ::3, tf.newaxis], peptide[:, 1::3, tf.newaxis], peptide[:, 2::3, tf.newaxis]], axis=-1)

    mapped_peptide_len_one = tf.squeeze(tf.gather(PeptideGenerator.len_one_subword_mapper, peptide_len_one, axis=-1))
    mapped_peptide_len_two = tf.gather_nd(PeptideGenerator.len_two_subword_mapper, peptide_len_two)
    mapped_peptide_len_three = tf.gather_nd(PeptideGenerator.len_three_subword_mapper, peptide_len_three)[:, :, tf.newaxis]
    mapped_peptide = tf.concat([mapped_peptide_len_one, mapped_peptide_len_two, mapped_peptide_len_three], axis=-1)

    return (mapped_peptide, mapped_allele), isGenerated  # model take peptide first


def build_pipeline(args: TrainOption, kfold: int = None) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    if args.dataset == 'MSI011320' or args.dataset == 'MSI011320_ANTI051821Z_COMBINE':
        dataset_cls = DATASET_TO_CLASS_FACTORY[args.dataset]
        bs_train_negative = int(args.batch_size_train * args.gen_neg_ratio)
        bs_train_positve = args.batch_size_train - bs_train_negative

        # initalize value (mainly dataset_size)
        dataset_cls.initialize_dry_run(kfold=kfold, experiment_name=args.experiment_name, phase='train', batch_size=bs_train_positve,
                                       remainder_method='sample', root_dir=args.root_dir, is_map_allele2index=True, random_state=3515)
        dataset_cls.initialize_dry_run(kfold=kfold, experiment_name=args.experiment_name, phase='eval', batch_size=args.batch_size_test,
                                       remainder_method='repeat', root_dir=args.root_dir, is_map_allele2index=True)
        dataset_cls.initialize_dry_run(kfold=kfold, experiment_name=args.experiment_name, phase='test', batch_size=args.batch_size_test,
                                       remainder_method='repeat', root_dir=args.root_dir, is_map_allele2index=True)

        # train_pipeline = dataset_cls(kfold=kfold, phase='train', root_dir=args.root_dir)
        # eval_pipeline need to trim the duplicate value to fill last batch
        eval_pipeline: DatasetType = dataset_cls.__new__(kfold=kfold, experiment_name=args.experiment_name, phase='eval', batch_size=args.batch_size_test,
                                            remainder_method='repeat', root_dir=args.root_dir, is_map_allele2index=True)
        test_pipeline: DatasetType = dataset_cls.__new__(kfold=kfold, experiment_name=args.experiment_name, phase='test', batch_size=args.batch_size_test,
                                            remainder_method='repeat', root_dir=args.root_dir, is_map_allele2index=True)

        train_partial_gen_negative_stage = partial(fill_nagative_stage, bs_train_positve, bs_train_negative)
        # train_partial_peptide_to_int32_stage = partial(peptide_to_int32_stage, bs_train_positve)
        # test_partial_peptide_to_int32_stage = partial(peptide_to_int32_stage, args.batch_size_test)

        # if you want to improve, you can add shuffle with reshuffle_each_iteration=True
        train_pipeline = tf.data.Dataset.range(args.epoch)\
            .interleave(lambda random_state:
                        dataset_cls.__new__(kfold=kfold, experiment_name=args.experiment_name, phase='train', batch_size=bs_train_positve, remainder_method='sample',
                                            root_dir=args.root_dir, is_map_allele2index=True, random_state=random_state)
                        .prefetch(tf.data.AUTOTUNE)
                        .batch(bs_train_positve)
                        .map(train_partial_gen_negative_stage)
                        .map(allele_to_int32_stage)
                        .map(index_to_subword_stage),
                        num_parallel_calls=2)\
            .prefetch(tf.data.AUTOTUNE)  # interleave should just be two (for this epoch and next epoch)

        eval_pipeline = eval_pipeline\
            .batch(args.batch_size_test)\
            .map(allele_to_int32_stage)\
            .map(index_to_subword_stage)\
            .repeat(args.epoch)\
            .prefetch(tf.data.AUTOTUNE)

        test_pipeline = test_pipeline\
            .batch(args.batch_size_test)\
            .map(allele_to_int32_stage)\
            .map(index_to_subword_stage)\
            .prefetch(tf.data.AUTOTUNE)

        # , num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    elif args.dataset == 'ANTI051821Z':
        pass  # TODO
        raise NotImplementedError('TODO')
    elif args.dataset == 'PRETRAIN_3D':
        WORD_LENGHT = 3
        logging.debug("start loading 3d pretrain central2context_path training data")
        central2context = YamlHelper.load_central2context(args.central2context_path)

        logging.debug("start loading 3d pretrain pair_map_counter_path training data")
        pair_map_counter = YamlHelper.load_pair_map_counter(args.pair_map_counter_path)

        word_map_couter = {pair: count for pair, count in pair_map_counter.items() if len(pair[0]) == len(pair[1]) == WORD_LENGHT}
        pair_map_distribution: np.ndarray = np.asarray(list(word_map_couter.values()))
        pair_map_distribution = pair_map_distribution/np.sum(pair_map_distribution)

        vocab_mhc_list = []
        for central, contexts in central2context.items():
            vocab_mhc_list.append(central)
            vocab_mhc_list.extend(contexts)
        vocab_mhc = set(vocab_mhc_list)
        vocab_mhc_word = [subword for subword in vocab_mhc if len(subword) == WORD_LENGHT]

        bs_train = args.batch_size_train
        bs_train_negative = int(args.batch_size_train * args.gen_neg_ratio)
        bs_train_positive = args.batch_size_train - bs_train_negative

        def create_subword(word, n_min=1, n_max=3):
            """
            Create subword

            # Example

            >> create_subword('abc')  

            ['a', 'ab', 'abc', 'b', 'bc', 'c']
            """
            subwords = []
            for j in range(len(word)):
                for i in range(n_min, n_max+1-j):
                    subwords.append(word[j:j+i])
            return subwords

        # TODO: convert all this inefficient code
        data_unmap_list = np.asarray(list(word_map_couter.keys()))
        data_list = [(create_subword(central), create_subword(context)) for central, context in data_unmap_list]
        train_data = np.asarray([([HLAProcessor.protein2int_mhc[c_subs_central] for c_subs_central in subs_central], [HLAProcessor.protein2int_mhc[c_subs_context]
                                for c_subs_context in subs_context]) for subs_central, subs_context in data_list], dtype=np.int)
        dataset_size = shared_memory.ShareableList(name=f'{TrainOption.dataset}_{TrainOption.experiment_name}_dataset_size')
        dataset_size[0] = train_data.shape[0]
        dataset_size.shm.close()  # written, then close

        def getTrainBatch():
            while(True):
                labels = np.zeros(shape=(bs_train), dtype=np.float32)
                data_index = np.random.choice(len(data_list), (bs_train_positive), p=pair_map_distribution)
                couples = train_data[data_index]
                labels[:bs_train_positive] = 1.0

                negative_couples = []
                for central_idx in random.sample(list(data_index), bs_train_negative):  # sample some central word to make its negative counter part
                    context_words = central2context[data_list[central_idx][0][2]]
                    negative_word = random.choice(vocab_mhc_word)
                    while(negative_word in context_words):
                        negative_word = random.choice(vocab_mhc_word)
                    negative_couples.append((train_data[central_idx][0], np.asarray([HLAProcessor.protein2int_mhc[c_subs_negative]
                                            for c_subs_negative in create_subword(negative_word)], dtype=np.int)))
                couples = np.append(couples, negative_couples, axis=0)
                yield (couples[:, 0, :], couples[:, 1, :]), labels

        return getTrainBatch(), None, None
    elif args.dataset == 'PRETRAIN_3D-HUMAN_PEPTIDE-TWO_HEAD':
        # Implement batch that output two things
        WORD_LENGHT = 3
        # TODO: implement PRETRAIN_3D generator function
        # load the prepared pretraining data
        logging.debug("start loading 3d pretrain central2context_path training data")
        central2context = YamlHelper.load_central2context(args.central2context_path)

        logging.debug("start loading 3d pretrain pair_map_counter_path training data")
        pair_map_counter = YamlHelper.load_pair_map_counter(args.pair_map_counter_path)
        logging.debug("converting 3d data to distribution")
        word_map_couter = {pair: count for pair, count in pair_map_counter.items() if len(pair[0]) == len(pair[1]) == WORD_LENGHT}
        pair_map_distribution: np.ndarray = np.asarray(list(word_map_couter.values()))
        pair_map_distribution = pair_map_distribution/np.sum(pair_map_distribution)

        logging.debug("start loading human pretrain human_peptide_central2context_path training data")
        human_peptide_central2context = YamlHelper.load_central2context(args.human_peptide_central2context_path)
        logging.debug("start loading human pretrain human_peptide_pair_map_counter_path training data")
        human_peptide_pair_map_counter = YamlHelper.load_pair_map_counter(args.human_peptide_pair_map_counter_path)
        logging.debug("converting human data to distribution")
        human_peptide_word_map_couter = {pair: count for pair, count in human_peptide_pair_map_counter.items() if len(pair[0]) == len(pair[1]) == WORD_LENGHT}
        human_peptide_pair_map_distribution: np.ndarray = np.asarray(list(human_peptide_word_map_couter.values()))
        # human_peptide_pair_map_distribution = human_peptide_pair_map_distribution/np.sum(human_peptide_pair_map_distribution) # we sum after we apply support_pair_mask

        vocab_mhc_list = []
        for central, contexts in central2context.items():
            vocab_mhc_list.append(central)
            vocab_mhc_list.extend(contexts)
        vocab_mhc = set(vocab_mhc_list)
        vocab_mhc_word = [subword for subword in vocab_mhc if len(subword) == WORD_LENGHT]

        logging.debug("creating vocab_human_peptide")
        vocab_human_peptide_list = []
        for central, contexts in human_peptide_central2context.items():
            vocab_human_peptide_list.append(central)
            vocab_human_peptide_list.extend(contexts)
        vocab_human_peptide = set(vocab_human_peptide_list)
        vocab_human_peptide_word = [subword for subword in vocab_human_peptide if len(subword) == WORD_LENGHT]
        # we use this to gen negative, need only need to get the one in word
        supported_vocab_human_peptide_word = list(set(HLAProcessor.protein2int_mhc.keys()).intersection(set(vocab_human_peptide_word)))

        def create_subword(word: str, n_min: int = 1, n_max: int = 3):
            """
            Create subword

            # Example

            >> create_subword('abc')  

            ['a', 'ab', 'abc', 'b', 'bc', 'c']
            """
            subwords: list[str] = []
            for j in range(len(word)):
                for i in range(n_min, n_max+1-j):
                    subwords.append(word[j:j+i])
            return subwords

        # TODO: convert all this inefficient code
        logging.debug("creating 3d training data for indexing")
        data_unmap_list = np.asarray(list(word_map_couter.keys()))
        data_list = [(create_subword(central), create_subword(context)) for central, context in data_unmap_list]
        train_data = np.asarray([([HLAProcessor.protein2int_mhc[c_subs_central] for c_subs_central in subs_central], [HLAProcessor.protein2int_mhc[c_subs_context]
                                for c_subs_context in subs_context]) for subs_central, subs_context in data_list], dtype=np.int)
        dataset_size = shared_memory.ShareableList(name=f'{TrainOption.dataset}_{TrainOption.experiment_name}_dataset_size')
        dataset_size[0] = train_data.shape[0]
        dataset_size.shm.close()  # written, then close
        logging.debug("finish creating 3d training data for indexing")

        # for human_peptide_ we considered it as suppliment, so it doens't get to be the main number of training data
        # we can't make human_peptide_data_list, it's too huge
        # we have to filter word that is supported only, TODO: detemine if we take HLA side or the protein side
        # supported_human_peptide_word_unmap = set(HLAProcessor.protein2int_mhc.keys()).intersection(set(human_peptide_word_map_couter.keys()))
        human_peptide_data_unmap_list = np.asarray(list(human_peptide_word_map_couter.keys()))
        support_pair_mask = np.array([(central in HLAProcessor.protein2int_mhc and context in HLAProcessor.protein2int_mhc) for (central, context) in human_peptide_word_map_couter.keys()])
        # now we filter only the supported one, resulted in 2939517/16008142 (18.36%)
        human_peptide_data_unmap_list = human_peptide_data_unmap_list[support_pair_mask]
        human_peptide_pair_map_distribution = human_peptide_pair_map_distribution[support_pair_mask]
        human_peptide_pair_map_distribution = human_peptide_pair_map_distribution / np.sum(human_peptide_pair_map_distribution)  # prop has to sum to one
        # human_peptide_data_list = [(create_subword(central), create_subword(context)) for central, context in human_peptide_data_unmap_list]
        # human_peptide_train_data = np.asarray([([HLAProcessor.protein2int_mhc[c_subs_central] for c_subs_central in subs_central], [HLAProcessor.protein2int_mhc[c_subs_context]
        #                                                                                                                             for c_subs_context in subs_context]) for subs_central, subs_context in human_peptide_data_list], dtype=np.int)

        bs_train = args.batch_size_train
        bs_train_negative = int(args.batch_size_train * args.gen_neg_ratio)
        bs_train_negative_3d, bs_train_negative_human = round(bs_train_negative/2), bs_train_negative - round(bs_train_negative/2)
        bs_train_positive = args.batch_size_train - bs_train_negative
        bs_train_positive_3d, bs_train_positive_human = round(bs_train_positive/2), bs_train_positive - round(bs_train_positive/2)

        def getTrainBatch():
            while(True):
                # left is 3d, right is human peptide
                labels = [np.zeros(shape=(bs_train), dtype=np.float32), np.zeros(shape=(bs_train), dtype=np.float32)]
                # 3d turn
                data_index_3d = np.random.choice(len(data_list), (bs_train_positive_3d), p=pair_map_distribution)
                couples = train_data[data_index_3d]
                # human peptide turn
                data_index_human = np.random.choice(len(human_peptide_data_unmap_list), (bs_train_positive_human), p=human_peptide_pair_map_distribution)
                # human_couples = [(create_subword(central), create_subword(context)) for (central, context) in human_peptide_data_unmap_list[data_index_human]]
                human_couples = [([HLAProcessor.protein2int_mhc[c_subs_central] for c_subs_central in create_subword(central)], [HLAProcessor.protein2int_mhc[c_subs_context]
                                  for c_subs_context in create_subword(context)]) for (central, context) in human_peptide_data_unmap_list[data_index_human]]
                couples = np.append(couples, human_couples, axis=0)

                labels[0][:bs_train_positive] = 1.0
                labels[1][:bs_train_positive] = 1.0

                # gen negative from 3d
                negative_couples = []
                for negative_offset, central_idx in enumerate(random.sample(list(data_index_3d), bs_train_negative_3d)):  # sample some central word to make its negative counter part
                    context_words = central2context[data_list[central_idx][0][2]]
                    negative_word = random.choice(vocab_mhc_word)
                    while(negative_word in context_words):
                        negative_word = random.choice(vocab_mhc_word)
                    negative_couples.append((train_data[central_idx][0], np.asarray([HLAProcessor.protein2int_mhc[c_subs_negative]
                                            for c_subs_negative in create_subword(negative_word)], dtype=np.int)))
                    # check if this negative is positive for human
                    if ((human_context := (human_peptide_central2context.get(data_list[central_idx][0][2], None))) is not None) and negative_word in human_context:
                        labels[1][bs_train_positive + negative_offset] = 1.0

                for negative_offset, human_central_idx in enumerate(random.sample(list(data_index_human), bs_train_negative_human)):  # sample some central word to make its negative counter part
                    human_central_word = human_peptide_data_unmap_list[human_central_idx][0]
                    human_context_words = human_peptide_central2context[human_central_word]
                    human_negative_word = random.choice(supported_vocab_human_peptide_word)
                    while(human_negative_word in human_context_words):
                        human_negative_word = random.choice(supported_vocab_human_peptide_word)
                    negative_couples.append((np.asarray([HLAProcessor.protein2int_mhc[c_subs_positive] for c_subs_positive in create_subword(human_central_word)], dtype=np.int),
                                             np.asarray([HLAProcessor.protein2int_mhc[c_subs_negative] for c_subs_negative in create_subword(human_negative_word)], dtype=np.int),))
                    # check if this negative is positive for 3d
                    if ((context_3d := (central2context.get(human_central_word, None))) is not None) and human_negative_word in context_3d:
                        labels[0][bs_train_positive + bs_train_negative_3d + negative_offset] = 1.0
                # now is time for negative_couples to be filled from the human peptide
                couples = np.append(couples, negative_couples, axis=0)

                yield (couples[:, 0, :], couples[:, 1, :]), labels
                # yield (tf.convert_to_tensor(couples[:, 0, :], dtype=tf.int64), tf.convert_to_tensor(couples[:, 1, :], dtype=tf.int64),), (tf.convert_to_tensor(labels[0], dtype=tf.float32), tf.convert_to_tensor(labels[1], dtype=tf.float32),)

        return getTrainBatch(), None, None

    elif args.dataset == 'PRETRAIN_3D-HUMAN_PEPTIDE-TWO_HEAD_DATASET_VERSION':
        # I try convert it into dataset, as I thought that it could get around GIL, but the CPU Utilize is just 140%, so, not working as expected
        logging.warning("Using dataset is not used by the official implementation")
        # Implement batch that output two things
        WORD_LENGHT = 3
        # TODO: implement PRETRAIN_3D generator function
        rng = np.random.RandomState(3515)
        # load the prepared pretraining data
        logging.debug("start loading 3d pretrain central2context_path training data")
        central2context = YamlHelper.load_central2context(args.central2context_path)

        logging.debug("start loading 3d pretrain pair_map_counter_path training data")
        pair_map_counter = YamlHelper.load_pair_map_counter(args.pair_map_counter_path)
        logging.debug("converting 3d data to distribution")
        word_map_couter = {pair: count for pair, count in pair_map_counter.items() if len(pair[0]) == len(pair[1]) == WORD_LENGHT}
        pair_map_distribution: np.ndarray = np.asarray(list(word_map_couter.values()))
        pair_map_distribution = pair_map_distribution/np.sum(pair_map_distribution)

        logging.debug("start loading human pretrain human_peptide_central2context_path training data")
        human_peptide_central2context = YamlHelper.load_central2context(args.human_peptide_central2context_path)
        logging.debug("start loading human pretrain human_peptide_pair_map_counter_path training data")
        human_peptide_pair_map_counter = YamlHelper.load_pair_map_counter(args.human_peptide_pair_map_counter_path)
        logging.debug("converting human data to distribution")
        human_peptide_word_map_couter = {pair: count for pair, count in human_peptide_pair_map_counter.items() if len(pair[0]) == len(pair[1]) == WORD_LENGHT}
        human_peptide_pair_map_distribution: np.ndarray = np.asarray(list(human_peptide_word_map_couter.values()))
        # human_peptide_pair_map_distribution = human_peptide_pair_map_distribution/np.sum(human_peptide_pair_map_distribution) # we sum after we apply support_pair_mask

        vocab_mhc_list = []
        for central, contexts in central2context.items():
            vocab_mhc_list.append(central)
            vocab_mhc_list.extend(contexts)
        vocab_mhc = set(vocab_mhc_list)
        vocab_mhc_word = [subword for subword in vocab_mhc if len(subword) == WORD_LENGHT]

        logging.debug("creating vocab_human_peptide")
        vocab_human_peptide_list = []
        for central, contexts in human_peptide_central2context.items():
            vocab_human_peptide_list.append(central)
            vocab_human_peptide_list.extend(contexts)
        vocab_human_peptide = set(vocab_human_peptide_list)
        vocab_human_peptide_word = [subword for subword in vocab_human_peptide if len(subword) == WORD_LENGHT]
        # we use this to gen negative, need only need to get the one in word
        supported_vocab_human_peptide_word = list(set(HLAProcessor.protein2int_mhc.keys()).intersection(set(vocab_human_peptide_word)))

        def create_subword(word: str, n_min: int = 1, n_max: int = 3):
            """
            Create subword

            # Example

            >> create_subword('abc')  

            ['a', 'ab', 'abc', 'b', 'bc', 'c']
            """
            subwords: list[str] = []
            for j in range(len(word)):
                for i in range(n_min, n_max+1-j):
                    subwords.append(word[j:j+i])
            return subwords

        # TODO: convert all this inefficient code
        logging.debug("creating 3d training data for indexing")
        data_unmap_list = np.asarray(list(word_map_couter.keys()))
        data_list = [(create_subword(central), create_subword(context)) for central, context in data_unmap_list]
        train_data = np.asarray([([HLAProcessor.protein2int_mhc[c_subs_central] for c_subs_central in subs_central], [HLAProcessor.protein2int_mhc[c_subs_context]
                                for c_subs_context in subs_context]) for subs_central, subs_context in data_list], dtype=np.int)
        dataset_size = shared_memory.ShareableList(name=f'{TrainOption.dataset}_{TrainOption.experiment_name}_dataset_size')
        dataset_size[0] = train_data.shape[0]
        dataset_size.shm.close()  # written, then close
        logging.debug("finish creating 3d training data for indexing")

        # for human_peptide_ we considered it as suppliment, so it doens't get to be the main number of training data
        # we can't make human_peptide_data_list, it's too huge
        # we have to filter word that is supported only, TODO: detemine if we take HLA side or the protein side
        # supported_human_peptide_word_unmap = set(HLAProcessor.protein2int_mhc.keys()).intersection(set(human_peptide_word_map_couter.keys()))
        human_peptide_data_unmap_list = np.asarray(list(human_peptide_word_map_couter.keys()))
        support_pair_mask = np.array([(central in HLAProcessor.protein2int_mhc and context in HLAProcessor.protein2int_mhc) for (central, context) in human_peptide_word_map_couter.keys()])
        # now we filter only the supported one, resulted in 2939517/16008142 (18.36%)
        human_peptide_data_unmap_list = human_peptide_data_unmap_list[support_pair_mask]
        human_peptide_pair_map_distribution = human_peptide_pair_map_distribution[support_pair_mask]
        human_peptide_pair_map_distribution = human_peptide_pair_map_distribution / np.sum(human_peptide_pair_map_distribution)  # prop has to sum to one
        # human_peptide_data_list = [(create_subword(central), create_subword(context)) for central, context in human_peptide_data_unmap_list]
        # human_peptide_train_data = np.asarray([([HLAProcessor.protein2int_mhc[c_subs_central] for c_subs_central in subs_central], [HLAProcessor.protein2int_mhc[c_subs_context]
        #                                                                                                                             for c_subs_context in subs_context]) for subs_central, subs_context in human_peptide_data_list], dtype=np.int)

        bs_train = args.batch_size_train
        bs_train_negative = int(args.batch_size_train * args.gen_neg_ratio)
        bs_train_negative_3d, bs_train_negative_human = round(bs_train_negative/2), bs_train_negative - round(bs_train_negative/2)
        bs_train_positive = args.batch_size_train - bs_train_negative
        bs_train_positive_3d, bs_train_positive_human = round(bs_train_positive/2), bs_train_positive - round(bs_train_positive/2)

        def getTrainBatch(seed: int):
            base_seed = rng.randint(0, 10*train_data.shape[0])
            seed += base_seed
            batch_rng = np.random.RandomState(seed)
            # left is 3d, right is human peptide
            labels = [np.zeros(shape=(bs_train), dtype=np.float32), np.zeros(shape=(bs_train), dtype=np.float32)]
            # 3d turn
            data_index_3d = batch_rng.choice(len(data_list), (bs_train_positive_3d), p=pair_map_distribution)
            couples = train_data[data_index_3d]
            # human peptide turn
            data_index_human = batch_rng.choice(len(human_peptide_data_unmap_list), (bs_train_positive_human), p=human_peptide_pair_map_distribution)
            # human_couples = [(create_subword(central), create_subword(context)) for (central, context) in human_peptide_data_unmap_list[data_index_human]]
            human_couples = [([HLAProcessor.protein2int_mhc[c_subs_central] for c_subs_central in create_subword(central)], [HLAProcessor.protein2int_mhc[c_subs_context]
                                                                                                                             for c_subs_context in create_subword(context)]) for (central, context) in human_peptide_data_unmap_list[data_index_human]]
            couples = np.append(couples, human_couples, axis=0)

            labels[0][:bs_train_positive] = 1.0
            labels[1][:bs_train_positive] = 1.0

            # gen negative from 3d
            negative_couples = []
            for negative_offset, central_idx in enumerate(batch_rng.choice(list(data_index_3d), bs_train_negative_3d)):  # sample some central word to make its negative counter part
                context_words = central2context[data_list[central_idx][0][2]]
                negative_word = batch_rng.choice(vocab_mhc_word, 1)[0]
                while(negative_word in context_words):
                    negative_word = batch_rng.choice(vocab_mhc_word, 1)[0]
                negative_couples.append((train_data[central_idx][0], np.asarray([HLAProcessor.protein2int_mhc[c_subs_negative]
                                        for c_subs_negative in create_subword(negative_word)], dtype=np.int)))
                # check if this negative is positive for human
                if ((human_context := (human_peptide_central2context.get(data_list[central_idx][0][2], None))) is not None) and negative_word in human_context:
                    labels[1][bs_train_positive + negative_offset] = 1.0

            for negative_offset, human_central_idx in enumerate(batch_rng.choice(list(data_index_human), bs_train_negative_human)):  # sample some central word to make its negative counter part
                human_central_word = human_peptide_data_unmap_list[human_central_idx][0]
                human_context_words = human_peptide_central2context[human_central_word]
                human_negative_word = batch_rng.choice(supported_vocab_human_peptide_word, 1)[0]
                while(human_negative_word in human_context_words):
                    human_negative_word = batch_rng.choice(supported_vocab_human_peptide_word, 1)[0]
                negative_couples.append((np.asarray([HLAProcessor.protein2int_mhc[c_subs_positive] for c_subs_positive in create_subword(human_central_word)], dtype=np.int),
                                        np.asarray([HLAProcessor.protein2int_mhc[c_subs_negative] for c_subs_negative in create_subword(human_negative_word)], dtype=np.int),))
                # check if this negative is positive for 3d
                if ((context_3d := (central2context.get(human_central_word, None))) is not None) and human_negative_word in context_3d:
                    labels[0][bs_train_positive + bs_train_negative_3d + negative_offset] = 1.0
            # now is time for negative_couples to be filled from the human peptide
            couples = np.append(couples, negative_couples, axis=0)
            return couples[:, 0, :], couples[:, 1, :], *labels

        # def getTrainFunctionGenerator(seed: int):
        #     base_seed = rng.randint(0, 10*train_data.shape[0], size=train_data.shape[0])
        #     return partial(getTrainBatch, base_seed + seed)

        def tf_map_to_multiple_input(seed):
            """
            tf.py_function not support nested
            """
            # in1, in2, labels = tf.py_function(func=getTrainFunctionGenerator,
            #                                   inp=[seed],
            #                                   Tout=[tf.int32, tf.int32, tf.float32]
            #                                   )

            # auto graph failed to read type annotation, sad.
            # in1: tf.Tensor
            # in2: tf.Tensor
            # label1: tf.Tensor
            # label2: tf.Tensor
            in1, in2, label1, label2 = tf.py_function(func=getTrainBatch,
                                                      inp=[seed],
                                                      #  Tout=[tf.int32, tf.int32, tf.float32, tf.float32]
                                                      Tout=[
                                                          tf.TensorSpec(shape=(bs_train, 6,), dtype=tf.int32),  # 6 is hard code because we only support word of length 3
                                                          tf.TensorSpec(shape=(bs_train, 6,), dtype=tf.int32),
                                                          tf.TensorSpec(shape=(bs_train,), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(bs_train,), dtype=tf.float32),
                                                      ],
                                                      )

            in1.set_shape([bs_train, 6])
            in2.set_shape([bs_train, 6])
            label1.set_shape([bs_train])
            label2.set_shape([bs_train])

            return (in1, in2), (label1, label2)

        train_dataset = tf.data.Dataset.from_generator(lambda: list(range(ceil(train_data.shape[0] + 1 / TrainOption.batch_size_train))), tf.int64)
        train_dataset = train_dataset.map(tf_map_to_multiple_input,
                                          num_parallel_calls=tf.data.AUTOTUNE,
                                          )
        # train_dataset.map(tf_map_to_multiple_input)
        # it has already return a batch

        return train_dataset, None, None
    else:
        raise RuntimeError("Unable to build such pipeline")
    return train_pipeline, eval_pipeline, test_pipeline


def build_custom_csv_dataset(args: MHCToolOption):

    # /media/zen3515/Data/Zenthesis/datasets/ANTI051821Z/antigen_information_051821_rev1_processed.csv
    csv_home_dir, csv_fname = os.path.split(os.path.abspath(args.CSV_PATH))

    class CustomCSVDataset(CSVDataset):
        home_dir = csv_home_dir
        include_unknown_allele = not args.IGNORE_UNKNOW  # TODO: make the parser to load the known allele first

        @classmethod
        def get_csv_path(cls, kfold: int, phase: typing.Literal['train', 'eval', 'test'], root_dir: typing.Optional[str] = None):
            return args.CSV_PATH

        @classmethod
        def save_result(cls, save_path: str, prediction: np.ndarray, batch_size=256):
            csv_path = cls.get_csv_path(None, None, None).encode('utf-8')
            raw_save_data = [(*row, pred) for pred, row in zip(np.squeeze(prediction), cls._generator(0, csv_path, 'inference', 'test',
                                                                                                      batch_size, remainder_method='repeat',
                                                                                                      is_map_allele2index=False, is_map_peptide2index=False, random_state=-1, will_log=False))]
            last_row = raw_save_data[-1]
            last_index = len(raw_save_data) - 1
            for i in range(2, batch_size):
                if last_row != raw_save_data[-i]:
                    last_index = len(raw_save_data) - i + 1
                    break
            # trim it
            raw_save_data = raw_save_data[:last_index + 1]
            save_df = pd.DataFrame(raw_save_data, columns=['Allele', 'Peptide', 'isGenerated', 'Prediction'])
            save_df.to_csv(save_path)

    return CustomCSVDataset


def build_inference_pipeline(args: MHCToolOption, batch_size=256):
    # TODO: move batch size somewhere?
    dataset_cls = build_custom_csv_dataset(args)
    dataset_cls.initialize_dry_run(kfold=0, experiment_name='inference', phase='test', batch_size=batch_size,
                                   remainder_method='repeat', root_dir='', is_map_allele2index=True)
    # TODO: why dataset_cls() is not working, has to use dataset_cls.__new__() instead? check if train is still working
    inference_pipeline: DatasetType = dataset_cls.__new__(kfold=0, experiment_name='inference', phase='test', batch_size=batch_size,
                                             remainder_method='repeat', root_dir='', is_map_allele2index=True)
    inference_pipeline = inference_pipeline\
        .batch(batch_size)\
        .map(allele_to_int32_stage)\
        .map(index_to_subword_stage)\
        .prefetch(tf.data.AUTOTUNE)
    return dataset_cls, inference_pipeline
