import os
import tensorflow as tf

def tensorboard(experiment_dir):
    tb_dir = os.path.join(experiment_dir, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
        
    return tf.keras.callbacks.TensorBoard(
        log_dir=tb_dir,
        profile_batch=0,
        histogram_freq=1,
    )

def early_stopping(patience):
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience,
        restore_best_weights=True,
    )

def checkpoints(experiment_dir):
    ckpt_dir = os.path.join(experiment_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
        save_weights_only=True,
    )

def save_best(experiment_dir):
    best_dir = os.path.join(experiment_dir, 'best')
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(best_dir, 'model'), 
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
    )