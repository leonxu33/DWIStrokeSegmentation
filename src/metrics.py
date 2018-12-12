from keras import backend as K
import tensorflow as tf

def mean_iou(label, prediction):
    prediction_ = tf.to_int32(prediction > 0.5)
    score, conf_matrix = tf.metrics.mean_iou(label, prediction_, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([conf_matrix]):
        score = tf.identity(score)
    return score

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + 1.)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


