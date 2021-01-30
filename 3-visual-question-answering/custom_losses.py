import tensorflow as tf
import tensorflow.keras.backend as K
import sys

# Categorical crossentropy weighted by a cost matrix of true label x predicted label
class WeightedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    
    def __init__(self, cost_mat, name='weighted_categorical_crossentropy', **kwargs):
        assert cost_mat.ndim == 2
        assert cost_mat.shape[0] == cost_mat.shape[1]
        
        super().__init__(name=name, **kwargs)
        self.cost_mat = K.cast_to_floatx(cost_mat)
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        assert sample_weight is None, "should only be derived from the cost matrix"
      
        return super().__call__(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=get_sample_weights(y_true, y_pred, self.cost_mat),
        )


def get_sample_weights(y_true, y_pred, cost_m):
    num_classes = len(cost_m)

    y_pred.shape.assert_has_rank(2)
    y_pred.shape[1:].assert_is_compatible_with(num_classes)
    y_pred.shape.assert_is_compatible_with(y_true.shape)

    y_pred = K.one_hot(K.argmax(y_pred), num_classes)

    y_true_nk1 = K.expand_dims(y_true, 2)
    y_pred_n1k = K.expand_dims(y_pred, 1)
    cost_m_1kk = K.expand_dims(cost_m, 0)

    sample_weights_nkk = cost_m_1kk * y_true_nk1 * y_pred_n1k
    sample_weights_n = K.sum(sample_weights_nkk, axis=[1, 2])

    return sample_weights_n

# Based on "Semantically Guided Visual Question Answering", 2018
def WAVCategoricalCrossentropy(l, embedding_matrix):
    emb_matrix = K.cast_to_floatx(embedding_matrix)
    emb_matrix = K.expand_dims(emb_matrix, 0)

    def loss(y_true, y_pred):
        t_vector = K.dot(y_true, emb_matrix)
        p_vector = K.dot(y_pred, emb_matrix)
        wav_term = l * tf.norm(p_vector - t_vector)
        return K.categorical_crossentropy(y_true, y_pred) + wav_term
        
    return loss

# Sparse version of WAVCategoricalCrossentropy
def WAVSparseCategoricalCrossentropy(l, embedding_matrix):
    emb_matrix = K.cast_to_floatx(embedding_matrix)
    emb_matrix = K.expand_dims(emb_matrix, 0)

    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='int32')  # cast needed for some reason, even though the output of the dataset is int32
        t_vector = tf.gather_nd(emb_matrix, y_true)
        p_vector = K.dot(y_pred, emb_matrix)
        wav_term = l * tf.norm(p_vector - t_vector)
        return K.sparse_categorical_crossentropy(y_true, y_pred) + wav_term
        
    return loss
