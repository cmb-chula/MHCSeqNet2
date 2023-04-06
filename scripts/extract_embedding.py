import sys  # nopep8
sys.path.append('.')  # nopep8

from core.models import MHCSeqNet2

import os
import numpy as np
import tensorflow as tf

MODEL_WEIGHT_PATH = 'resources/trained_weight/embedding-3d/GloVeFastText.h5'

if __name__ == '__main__':
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

    model = MHCSeqNet2.buildModel()
    model.load_weights(MODEL_WEIGHT_PATH)
    peptide_embeddings_matrix: np.ndarray = np.copy(model.get_layer('peptide_embeddings').get_weights()[0])
    allele_embeddings_matrix: np.ndarray = np.copy(model.get_layer('allele_embeddings').get_weights()[0])
    peptide_embeddings_matrix_path = os.path.join(os.path.split(MODEL_WEIGHT_PATH)[0], 'peptide_embeddings_matrix.npy')
    allele_embeddings_matrix_path = os.path.join(os.path.split(MODEL_WEIGHT_PATH)[0], 'allele_embeddings_matrix.npy')
    np.save(peptide_embeddings_matrix_path, peptide_embeddings_matrix)
    np.save(allele_embeddings_matrix_path, allele_embeddings_matrix)
    print(f"saved embedding at:\n{peptide_embeddings_matrix_path}\n{allele_embeddings_matrix_path}")
