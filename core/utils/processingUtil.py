# from __future__ import annotations

import typing
import os
import yaml

from glob import glob

# if typing.TYPE_CHECKING:
import tensorflow as tf
import numpy as np


class HLAProcessor:
    is_loaded: bool = False
    mapper: typing.Dict[str, str] = None
    mapper2index: typing.Dict[str, int] = None
    protein2int_mhc: typing.Dict[str, int] = None
    int2protein_mhc: typing.Dict[int, str] = None
    allele_index2index_list: tf.Tensor = None
    len_one_subword_allele_mapper: tf.Tensor = None
    len_two_subword_allele_mapper: tf.Tensor = None
    len_three_subword_allele_mapper: tf.Tensor = None
    ALLELE_LEN_ONE_MAPPER_LIST: typing.List[str] = ['-', '?', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    ALLELE_LEN_ONE_MAPPER_DICT: typing.Dict[str, int] = None

    # @classmethod
    # def map_hla2peptide_eager(cls, allele):
    #     return cls.mapper[allele.numpy().decode('utf-8')]

    # @classmethod
    # def map_hla2peptide(cls, allele: str):
    #     return cls.mapper[allele]

    @classmethod
    def map_allele_index2int(cls, allele_indice: tf.Tensor):
        """
        Take batch of allele in int32, convert to batch of list of int32
        """
        return tf.gather(cls.allele_index2index_list, allele_indice)

    @classmethod
    def load_from_disk(cls, vocab_allele_path: str, mapping_path: str):
        cls.mapper = {}
        for yaml_mapping_path in sorted(glob(os.path.join(mapping_path, '*.yaml'))):
            with open(yaml_mapping_path, 'r') as fileHandler:
                cls.mapper.update(yaml.safe_load(fileHandler))
        cls.mapper2index = {allele: index for index, allele in enumerate(sorted(cls.mapper.keys()))}
        with open(vocab_allele_path, 'r') as fileHandler:
            VOCAB_ALLELE = fileHandler.readline().strip().split(',')
        # This is the same as [subword for subword in VOCAB_ALLELE if len(subword) == 1] but I hard code it for consistency
        cls.ALLELE_LEN_ONE_MAPPER_LIST = ['-', '?', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        cls.ALLELE_LEN_ONE_MAPPER_DICT = {amino: index for index, amino in enumerate(cls.ALLELE_LEN_ONE_MAPPER_LIST)}
        len_one_mapper = len(cls.ALLELE_LEN_ONE_MAPPER_LIST)
        cls.protein2int_mhc = {p: i for i, p in enumerate(VOCAB_ALLELE)}
        cls.int2protein_mhc = {i: p for p, i in cls.protein2int_mhc.items()}

        missing_index = cls.ALLELE_LEN_ONE_MAPPER_DICT['?']
        allele_index2index_list = np.zeros(shape=(len(cls.mapper2index), 372,), dtype=np.int32) + missing_index  # ~ 7.2 MB
        for allele, allele_index in cls.mapper2index.items():
            int_seq = [cls.ALLELE_LEN_ONE_MAPPER_DICT[amino] for amino in cls.mapper[allele]]
            allele_index2index_list[allele_index] = int_seq
        cls.allele_index2index_list = tf.constant(allele_index2index_list, dtype=tf.int32)
        # usage tf.gather(len_one_subword_allele_mapper, cls.allele_index2index_list[0])

        missing_index = cls.protein2int_mhc['?']
        len_one_subword_allele_mapper = np.zeros((len_one_mapper,), np.int32) + missing_index
        for i, subword in enumerate(cls.ALLELE_LEN_ONE_MAPPER_LIST):
            mapped_index = cls.protein2int_mhc.get(subword, missing_index)
            len_one_subword_allele_mapper[i] = mapped_index
        cls.len_one_subword_allele_mapper = tf.constant(len_one_subword_allele_mapper, dtype=tf.int32)
        missing_index = cls.protein2int_mhc['??']
        len_two_subword_allele_mapper = np.zeros((len_one_mapper, len_one_mapper,), np.int32) + missing_index
        for i, first_a in enumerate(cls.ALLELE_LEN_ONE_MAPPER_LIST):
            for j, second_a in enumerate(cls.ALLELE_LEN_ONE_MAPPER_LIST):
                subword = f'{first_a}{second_a}'
                mapped_index = cls.protein2int_mhc.get(subword, missing_index)
                len_two_subword_allele_mapper[i, j] = mapped_index
            cls.len_two_subword_allele_mapper = tf.constant(len_two_subword_allele_mapper, dtype=tf.int32)
        missing_index = cls.protein2int_mhc['???']
        len_three_subword_allele_mapper = np.zeros((len_one_mapper, len_one_mapper, len_one_mapper,), np.int32) + missing_index
        for i, first_a in enumerate(cls.ALLELE_LEN_ONE_MAPPER_LIST):
            for j, second_a in enumerate(cls.ALLELE_LEN_ONE_MAPPER_LIST):
                for k, third_a in enumerate(cls.ALLELE_LEN_ONE_MAPPER_LIST):
                    subword = f'{first_a}{second_a}{third_a}'
                    mapped_index = cls.protein2int_mhc.get(subword, missing_index)
                    len_three_subword_allele_mapper[i, j, k] = mapped_index
                cls.len_three_subword_allele_mapper = tf.constant(len_three_subword_allele_mapper, dtype=tf.int32)
        cls.is_loaded = True


class PeptideGenerator:
    is_loaded: bool = False
    length_distribution_start_at: int = 3
    length_distribution: typing.Dict[int, float] = None
    log_length_distribution: tf.Tensor = None
    protein2int_peptide: typing.Dict[str, int] = None
    int2protein_peptide: typing.Dict[int, str] = None
    NEGATIVE_AMINO: typing.List[str] = None
    len_one_subword_mapper: tf.Tensor = None
    len_two_subword_mapper: tf.Tensor = None
    len_three_subword_mapper: tf.Tensor = None
    PEPTIDE_LEN_ONE_MAPPER_LIST: typing.List[str] = ['^',
                                                     'L', 'V', 'A', 'S', 'E', 'P', 'R', 'I', 'T', 'G', 'K', 'F', 'Y', 'Q', 'D', 'N', 'H', 'M', 'W', 'C',  # negative amino
                                                     'X', 'y', 'p', 's', 'v', 'n', 'B', 't', 'c', 'q', 'm']
    PEPTIDE_LEN_ONE_MAPPER_DICT: typing.Dict[str, int] = None
    PEPTIDE_LEN_ONE_MAPPER_TF: tf.lookup.StaticHashTable = None
    max_peptide_length: int = 45
    padding_index_value: int = 0

    @classmethod
    def generate_peptide(cls, generation_size: int) -> tf.Tensor:
        """
        Use gpu to speed up generation process
        """
        # generated_lenght need to plus 3 since the length_distribution start at 3 to 43
        generated_lenght = tf.random.categorical(cls.log_length_distribution, generation_size)[0] + cls.length_distribution_start_at
        generated_amino_index_value = tf.random.uniform(shape=(tf.math.reduce_sum(generated_lenght),), minval=1, maxval=len(cls.NEGATIVE_AMINO), dtype=tf.int32)
        generated_amino_index_value = generated_amino_index_value + 1  # need to shift one to align with PEPTIDE_LEN_ONE_MAPPER_LIST since it has ^ at the front
        generated_amino_index_full = tf.reverse(tf.RaggedTensor.from_row_lengths(values=generated_amino_index_value,
                                                row_lengths=generated_lenght).to_tensor(cls.padding_index_value, shape=(generation_size, cls.max_peptide_length)), axis=[-1])
        return generated_amino_index_full

    # @classmethod
    # def _generate_peptide_eager(cls, generated_lenght: tf.Tensor, generated_amino_index_full: tf.Tensor):
    #     generated_peptide = [''.join([cls.NEGATIVE_AMINO[amino] for amino in amino_list[:length]]) for length, amino_list in zip(generated_lenght, generated_amino_index_full)]
    #     return generated_peptide

    # @classmethod
    # def generate_lenght_from_distribution(cls, generation_size: int, allele: tf.Tensor):
    #     generated_lenght = tf.random.categorical(cls.log_length_distribution, generation_size)[0]
    #     generated_amino_index_full = tf.random.uniform(shape=(generation_size, tf.math.reduce_max(generated_lenght)), minval=1, maxval=len(cls.NEGATIVE_AMINO), dtype=tf.int32)
    #     generated_allele_index = tf.random.uniform(shape=(generation_size,), minval=1, maxval=allele.shape[0], dtype=tf.int8)
    #     return generated_lenght, generated_amino_index_full, generated_allele_index

    # @classmethod
    # def generate_peptide(cls, generated_lenght: tf.Tensor, generated_amino_index_full: tf.Tensor):
    #     return ["".join([cls.NEGATIVE_AMINO[amino] for amino in amino_list[:length]]) for length, amino_list in zip(generated_lenght, generated_amino_index_full)]

    @classmethod
    def map_peptide2int(cls, peptides: tf.Tensor):
        """
        Take batch of peptide in str convert to batch of list of int32
        """
        peptides = peptides.numpy().astype(f'<U{cls.max_peptide_length}')  # Little Endian, Unicode, 45 char length
        return tf.convert_to_tensor([[cls.padding_index_value]*(cls.max_peptide_length - len(peptide)) + [cls.PEPTIDE_LEN_ONE_MAPPER_DICT[amino] for amino in peptide] for peptide in peptides], dtype=tf.int32)

    @classmethod
    def map_peptide2int_graph(cls, peptides: tf.Tensor, num_peptide: int):
        """
        Take batch of peptide in str convert to batch of list of int32,  
        BUT THIS ONE DOESN'T NEED tf.py_function which allow parallel call, no more GIL
        WIP: unicode_split need to known inputsize right from the start, which means that we have to map it right from the start yield...()
        """
        raise NotImplementedError("WIP: unicode_split need to known inputsize right from the start, which means that we have to map it right from the start yield...()")
        ragged_mapped_peptide: tf.RaggedTensor = cls.PEPTIDE_LEN_ONE_MAPPER_TF[tf.strings.unicode_split(peptides, 'UTF-8')]
        return tf.reverse(ragged_mapped_peptide.to_tensor(cls.padding_index_value, shape=(num_peptide, cls.max_peptide_length)), axis=[-1])

    @classmethod
    def load_from_disk(cls, vocab_peptide_path: str, length_distribution_path: str, negative_amino_path: str):
        cls.length_distribution = {}
        cls.NEGATIVE_AMINO = []
        with open(length_distribution_path, 'r') as fileHandler:
            cls.length_distribution = yaml.safe_load(fileHandler)
            cls.log_length_distribution = tf.math.log([[prob for _lenght, prob in sorted(cls.length_distribution.items())]])
            cls.length_distribution_start_at = min(cls.length_distribution.keys())  # should be 3

        with open(negative_amino_path, 'r') as fileHandler:
            cls.NEGATIVE_AMINO = [amino.strip() for amino in fileHandler.read().strip().split(',')]
            # you can see that NEGATIVE_AMINO is PEPTIDE_LEN_ONE_MAPPER_LIST but shifted 1 from padding '^'

        with open(vocab_peptide_path, 'r') as fileHandler:
            VOCAB = fileHandler.readline().strip().split(',')
        # TODO: check what to do with unknown word, it becomes a problem when predicting swissprot data
        missing_index = -1
        cls.PEPTIDE_LEN_ONE_MAPPER_LIST = ['^', 'L', 'V', 'A', 'S', 'E', 'P', 'R', 'I', 'T', 'G', 'K', 'F',
                                           'Y', 'Q', 'D', 'N', 'H', 'M', 'W', 'C', 'X', 'y', 'p', 's', 'v', 'n', 'B', 't', 'c', 'q', 'm']
        cls.PEPTIDE_LEN_ONE_MAPPER_DICT = {amino: index for index, amino in enumerate(cls.PEPTIDE_LEN_ONE_MAPPER_LIST)}
        PEPTIDE_LEN_ONE_MAPPER_KEYS = tf.constant(cls.PEPTIDE_LEN_ONE_MAPPER_LIST)
        PEPTIDE_LEN_ONE_MAPPER_VALUES = tf.range(len(cls.PEPTIDE_LEN_ONE_MAPPER_DICT))
        cls.PEPTIDE_LEN_ONE_MAPPER_TF = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
            PEPTIDE_LEN_ONE_MAPPER_KEYS, PEPTIDE_LEN_ONE_MAPPER_VALUES), default_value=missing_index)
        len_one_mapper = len(cls.PEPTIDE_LEN_ONE_MAPPER_LIST)
        cls.protein2int_peptide = {p: i for i, p in enumerate(VOCAB)}
        cls.int2protein_peptide = {i: p for p, i in cls.protein2int_peptide.items()}
        len_one_subword_mapper = np.zeros((len_one_mapper,), np.int32) + missing_index
        for i, subword in enumerate(cls.PEPTIDE_LEN_ONE_MAPPER_LIST):
            mapped_index = cls.protein2int_peptide.get(subword, missing_index)
            len_one_subword_mapper[i] = mapped_index
        cls.len_one_subword_mapper = tf.constant(len_one_subword_mapper, dtype=tf.int32)
        # 302/1024 items are considered missing
        len_two_subword_mapper = np.zeros((len_one_mapper, len_one_mapper,), np.int32) + missing_index
        for i, first_a in enumerate(cls.PEPTIDE_LEN_ONE_MAPPER_LIST):
            for j, second_a in enumerate(cls.PEPTIDE_LEN_ONE_MAPPER_LIST):
                subword = f'{first_a}{second_a}'
                mapped_index = cls.protein2int_peptide.get(subword, missing_index)
                len_two_subword_mapper[i, j] = mapped_index
            cls.len_two_subword_mapper = tf.constant(len_two_subword_mapper, dtype=tf.int32)
        # 20764/32768 items are considered missing
        len_three_subword_mapper = np.zeros((len_one_mapper, len_one_mapper, len_one_mapper,), np.int32) + missing_index
        for i, first_a in enumerate(cls.PEPTIDE_LEN_ONE_MAPPER_LIST):
            for j, second_a in enumerate(cls.PEPTIDE_LEN_ONE_MAPPER_LIST):
                for k, third_a in enumerate(cls.PEPTIDE_LEN_ONE_MAPPER_LIST):
                    subword = f'{first_a}{second_a}{third_a}'
                    mapped_index = cls.protein2int_peptide.get(subword, missing_index)
                    len_three_subword_mapper[i, j, k] = mapped_index
                cls.len_three_subword_mapper = tf.constant(len_three_subword_mapper, dtype=tf.int32)
        cls.is_loaded = True
