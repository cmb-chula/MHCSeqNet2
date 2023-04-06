from tensorflow.keras.layers import Input, Embedding, Lambda, Activation, Dense, Dot
from tensorflow.keras import Model

import typing
import tensorflow as tf


class GloVeFastText:

    @classmethod
    def buildModel(cls,
                   subword_size: int = 6,
                   vocab_size: int = 3774,  # peptide_vocab_size: int = 12758, allele_vocab_size: int = 3774
                   dim_embeddings: int = 32,
                   classifier_activation: typing.Optional[str] = 'sigmoid',
                   ):

        central_input = Input(shape=(subword_size,), name='centralWord', dtype='int32')
        context_input = Input(shape=(subword_size,), name='contextWord', dtype='int32')
        embedded_sequences_central = Embedding(vocab_size, dim_embeddings, input_length=subword_size, trainable=True, name='central_embeddings')(central_input)
        embedded_sequences_context = Embedding(vocab_size, dim_embeddings, input_length=subword_size, trainable=True, name='context_embeddings')(context_input)
        embedded_sequences_central = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name='central_embeddings_sum')(embedded_sequences_central)
        embedded_sequences_context = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name='context_embeddings_sum')(embedded_sequences_context)

        merged_layer = Dot(axes=1)([embedded_sequences_central, embedded_sequences_context])
    #     merged_layer = Reshape((1,), input_shape=(1, 1))(merged_layer)
        output = Activation(activation=classifier_activation)(merged_layer)

        model = Model(inputs=[central_input, context_input], outputs=output)
        # model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=SGD(learning_rate=0.01))

        return model


class MultiHeadGloVeFastTextSplit:

    @classmethod
    def buildModel(cls,
                   subword_size: int = 6,
                   vocab_size: int = 3774,  # peptide_vocab_size: int = 12758, allele_vocab_size: int = 3774
                   dim_embeddings: int = 32,
                   classifier_activation: typing.Optional[str] = 'sigmoid',
                   head_names: typing.List[str] = ['3d', 'human-protien'],
                   ):

        assert dim_embeddings % len(head_names) == 0, "Number of heads doesn't match with embedding dimension"

        central_input = Input(shape=(subword_size,), name='centralWord', dtype='int32')
        context_input = Input(shape=(subword_size,), name='contextWord', dtype='int32')
        embedded_sequences_central = Embedding(vocab_size, dim_embeddings, input_length=subword_size, trainable=True, name='central_embeddings')(central_input)
        embedded_sequences_context = Embedding(vocab_size, dim_embeddings, input_length=subword_size, trainable=True, name='context_embeddings')(context_input)
        embedded_sequences_central = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name='central_embeddings_sum')(embedded_sequences_central)
        embedded_sequences_context = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name='context_embeddings_sum')(embedded_sequences_context)

        outputs = []
        # divide dimension of embedding by half per head
        paritioned_embedding_size = dim_embeddings//len(head_names)
        for ih, head_name in enumerate(head_names):
            partition_started = ih * paritioned_embedding_size
            merged_layer = Dot(axes=1)([embedded_sequences_central[:, partition_started: partition_started + paritioned_embedding_size],
                                        embedded_sequences_context[:, partition_started: partition_started + paritioned_embedding_size]])
            outputs.append(Activation(activation=classifier_activation, name=f"output_{head_name}")(merged_layer))

        model = Model(inputs=[central_input, context_input], outputs=outputs)

        return model


class MultiHeadGloVeFastTextJointed:
    """
    Using multiple head, but instead of seperating the dimension, we shared it and perform push-pull on dense feature instead
    """

    @classmethod
    def buildModel(cls,
                   subword_size: int = 6,
                   vocab_size: int = 3774,  # peptide_vocab_size: int = 12758, allele_vocab_size: int = 3774
                   dim_embeddings: int = 32,
                   classifier_activation: typing.Optional[str] = 'sigmoid',
                   head_names: typing.List[str] = ['3d', 'human-protien'],
                   ):

        assert dim_embeddings % len(head_names) == 0, "Number of heads doesn't match with embedding dimension"

        central_input = Input(shape=(subword_size,), name='centralWord', dtype='int32')
        context_input = Input(shape=(subword_size,), name='contextWord', dtype='int32')
        embedded_sequences_central = Embedding(vocab_size, dim_embeddings, input_length=subword_size, trainable=True, name='central_embeddings')(central_input)
        embedded_sequences_context = Embedding(vocab_size, dim_embeddings, input_length=subword_size, trainable=True, name='context_embeddings')(context_input)
        embedded_sequences_central = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name='central_embeddings_sum')(embedded_sequences_central)
        embedded_sequences_context = Lambda(lambda x: tf.math.reduce_sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name='context_embeddings_sum')(embedded_sequences_context)

        dim_dense_feature = dim_embeddings * len(head_names)
        embedded_sequences_central = Dense(dim_dense_feature, activation='relu', name='central_dense_feature')(embedded_sequences_central)
        embedded_sequences_context = Dense(dim_dense_feature, activation='relu', name='context_dense_feature')(embedded_sequences_context)

        outputs = []
        # divide dimension of embedding by half per head
        paritioned_embedding_size = dim_dense_feature//len(head_names)
        for ih, head_name in enumerate(head_names):
            partition_started = ih * paritioned_embedding_size
            merged_layer = Dot(axes=1)([embedded_sequences_central[:, partition_started: partition_started + paritioned_embedding_size],
                                        embedded_sequences_context[:, partition_started: partition_started + paritioned_embedding_size]])
            outputs.append(Activation(activation=classifier_activation, name=f"output_{head_name}")(merged_layer))

        model = Model(inputs=[central_input, context_input], outputs=outputs)

        return model


if __name__ == "__main__":
    # for debugging, to know the model structure
    from tensorflow.keras.utils import plot_model
    # model = MultiHeadGloVeFastTextJointed.buildModel()
    # plot_model(model, show_shapes=True, to_file='figures/MultiHeadGloVeFastTextJointed.png')
    model = GloVeFastText.buildModel()
    plot_model(model, show_shapes=True, to_file='figures/GloVeFastText.png')
