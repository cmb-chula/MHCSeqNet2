from tensorflow.keras.layers import Input, Embedding, Lambda, Dense, Dropout, Flatten, Concatenate, Bidirectional, GRU
from tensorflow.keras import Model

import typing
import tensorflow as tf


class MHCSeqNet2:

    @classmethod
    def buildModel(cls,
                   peptide_word_length: int = 15,
                   peptide_subword_size: int = 6,
                   allele_word_length: int = 124,
                   allele_subword_size: int = 6,
                   peptide_vocab_size: int = 12758,
                   allele_vocab_size: int = 3774,
                   trainable_peptide_embedding: bool = True,
                   trainable_allele_embedding: bool = True,
                   dim_embeddings: int = 32,
                   classifier_activation: typing.Optional[str] = 'sigmoid',
                   ):

        input1_peptide = Input(shape=(peptide_word_length, peptide_subword_size,))
        input2_allele = Input(shape=(allele_word_length, allele_subword_size,))

        embedded_sequences_peptide = Embedding(peptide_vocab_size, dim_embeddings, input_length=peptide_word_length,
                                               trainable=trainable_peptide_embedding, mask_zero=True, name='peptide_embeddings')(input1_peptide)
        embedded_sequences_peptide = Lambda(lambda x: tf.math.reduce_sum(x, axis=2), output_shape=lambda s: (
            s[1], s[2]), name='peptide_embeddings_sum')(embedded_sequences_peptide)

        embedded_sequences_allele = Embedding(allele_vocab_size, dim_embeddings, input_length=allele_word_length,
                                              trainable=trainable_allele_embedding, mask_zero=True, name='allele_embeddings')(input2_allele)
        embedded_sequences_allele = Lambda(lambda x: tf.math.reduce_sum(x, axis=2), output_shape=lambda s: (
            s[1], s[2]), name='allele_embeddings_sum')(embedded_sequences_allele)

        x_1 = Dense(250, activation='relu')(embedded_sequences_peptide)
        x_1 = Dropout(0.5)(x_1)
        x_1 = Flatten()(x_1)

        x_2 = Dense(250, activation='relu')(embedded_sequences_allele)
        x_2 = Dropout(0.5)(x_2)
        x_2 = Flatten()(x_2)

        x_3 = Concatenate()([x_1, x_2])
        x_3 = Dense(240, activation='relu')(x_3)
        x_3 = Dense(240, activation='relu')(x_3)
        x_3 = Dropout(0.4)(x_3)

        out = Dense(1, activation=classifier_activation)(x_3)

        model = Model(inputs=[input1_peptide, input2_allele], outputs=out)
        return model
        # model.compile(optimizer=SGD(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['acc'])


class MHCSeqNet2_GRUPeptide:

    @classmethod
    def buildModel(cls,
                   peptide_word_length: int = 15,
                   peptide_subword_size: int = 6,
                   peptide_gru_dim: int = 160,
                   peptide_dropout_gru: float = 0.4,
                   peptide_dropout_recurrence_gru: float = 0.3,
                   allele_word_length: int = 124,
                   allele_subword_size: int = 6,
                   peptide_vocab_size: int = 12758,
                   allele_vocab_size: int = 3774,
                   trainable_peptide_embedding: bool = True,
                   trainable_allele_embedding: bool = True,
                   dim_embeddings: int = 32,
                   classifier_activation: typing.Optional[str] = 'sigmoid',
                   ):

        input1_peptide = Input(shape=(peptide_word_length, peptide_subword_size,))
        input2_allele = Input(shape=(allele_word_length, allele_subword_size,))

        embedded_sequences_peptide = Embedding(peptide_vocab_size, dim_embeddings, input_length=peptide_word_length,
                                               trainable=trainable_peptide_embedding, mask_zero=True, name='peptide_embeddings')(input1_peptide)
        embedded_sequences_peptide = Lambda(lambda x: tf.math.reduce_sum(x, axis=2), output_shape=lambda s: (
            s[1], s[2]), name='peptide_embeddings_sum')(embedded_sequences_peptide)

        embedded_sequences_allele = Embedding(allele_vocab_size, dim_embeddings, input_length=allele_word_length,
                                              trainable=trainable_allele_embedding, mask_zero=True, name='allele_embeddings')(input2_allele)
        embedded_sequences_allele = Lambda(lambda x: tf.math.reduce_sum(x, axis=2), output_shape=lambda s: (
            s[1], s[2]), name='allele_embeddings_sum')(embedded_sequences_allele)

        # TODO: replace with GRU
        # x_1 = Dense(250, activation='relu')(embedded_sequences_peptide)
        x_1 = Bidirectional(GRU(peptide_gru_dim,
                                dropout=peptide_dropout_gru,
                                recurrent_dropout=peptide_dropout_recurrence_gru))(embedded_sequences_peptide)

        # x_1 = Dropout(0.5)(x_1) # GRU already has drop out
        x_1 = Flatten()(x_1)

        x_2 = Dense(250, activation='relu')(embedded_sequences_allele)
        x_2 = Dropout(0.5)(x_2)
        x_2 = Flatten()(x_2)

        x_3 = Concatenate()([x_1, x_2])
        x_3 = Dense(240, activation='relu')(x_3)
        x_3 = Dense(240, activation='relu')(x_3)
        x_3 = Dropout(0.4)(x_3)

        out = Dense(1, activation=classifier_activation)(x_3)

        model = Model(inputs=[input1_peptide, input2_allele], outputs=out)
        return model
