from core.utils.processingUtil import HLAProcessor
from core.utils import PeptideGenerator
from core.options import TrainOption
from .mhcseqnet2 import MHCSeqNet2, MHCSeqNet2_GRUPeptide
from .pretraining_model import GloVeFastText, MultiHeadGloVeFastTextSplit, MultiHeadGloVeFastTextJointed

import typing
import logging
import numpy as np

ARCHITECTURE_MAPPER: typing.Dict[typing.Type[TrainOption.MODEL_TYPE], typing.Union[MHCSeqNet2, MHCSeqNet2_GRUPeptide]] = {
    'MHCSeqNet2': MHCSeqNet2,
    'MHCSeqNet2_GRUPeptide': MHCSeqNet2_GRUPeptide,
    'GloVeFastText': GloVeFastText,
    'MultiHeadGloVeFastTextSplit': MultiHeadGloVeFastTextSplit,
    'MultiHeadGloVeFastTextJointed': MultiHeadGloVeFastTextJointed,
}


def load_matrix(new_vocab_length: int, embedding_matrix_path: str) -> np.ndarray:
    embedding_matrix = np.load(embedding_matrix_path)
    vocab_len_original: int
    embedding_dim: int
    vocab_len_original, embedding_dim = embedding_matrix.shape  # (9056, 32) and (3722, 32)
    random_init_padding = np.random.uniform(size=(new_vocab_length-vocab_len_original, embedding_dim))
    embedding_matrix = np.pad(embedding_matrix, ((0, new_vocab_length-vocab_len_original), (0, 0)), mode='constant', constant_values=0)
    embedding_matrix[vocab_len_original:] = random_init_padding
    return embedding_matrix


def build_model(args: TrainOption = None):
    """
    Inference stage got args=None
    """
    model_class = ARCHITECTURE_MAPPER[args.MODEL_TYPE]
    model = model_class.buildModel()
    if args is not None:
        if 'load_embedding_peptide' in dir(args) and args.load_embedding_peptide:
            peptide_embedding_matrix = load_matrix(len(PeptideGenerator.protein2int_peptide), args.embedding_peptide_path)
            logging.info(f"Loaded pretrain peptide embedding with shape={peptide_embedding_matrix.shape}")
            model.get_layer('peptide_embeddings').set_weights([np.copy(peptide_embedding_matrix)])
        if 'load_embedding_allele' in dir(args) and args.load_embedding_allele:
            allele_embedding_matrix = load_matrix(len(HLAProcessor.protein2int_mhc), args.embedding_allele_path)
            logging.info(f"Loaded pretrain allele embedding with shape={allele_embedding_matrix.shape}")
            model.get_layer('allele_embeddings').set_weights([np.copy(allele_embedding_matrix)])
    return model
