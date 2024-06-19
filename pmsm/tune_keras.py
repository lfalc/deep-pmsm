import keras_tuner
from preprocessing.data import LightDataManager
import preprocessing.config as cfg
import numpy as np
import tune_keras_class as tkc
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import time

# Ensure reproducibility
SEED = cfg.data_cfg["random_seed"]
np.random.seed(SEED)

dm = LightDataManager(cfg.data_cfg["file_path"])
dm.featurize()

# remove batch_size from dict, as it is not needed for build function
batch_size = 32
window_size = 32

x_train = dm.tra_df[dm.x_cols + [dm.PROFILE_ID_COL]]
y_train = dm.tra_df[dm.y_cols]
x_val = dm.val_df[dm.x_cols + [dm.PROFILE_ID_COL]]
y_val = dm.val_df[dm.y_cols]
x_tst = dm.tst_df[dm.x_cols + [dm.PROFILE_ID_COL]]
y_tst = dm.tst_df[dm.y_cols]

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        min_delta=1e2,
        patience=cfg.keras_cfg["early_stop_patience"],
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="loss", patience=cfg.keras_cfg["early_stop_patience"] // 3
    ),
    ModelCheckpoint(
        filepath='keras_tune/best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    TensorBoard(log_dir='keras_tune/logs')
]

LOG_DIR = f"{int(time.time())}"

tuner = keras_tuner.RandomSearch(
    hypermodel=tkc.PmsmHyperModel(),
    objective='val_loss',
    max_trials=2,
    executions_per_trial=1,
    overwrite=True,
    directory=LOG_DIR,
    project_name='cnn',)

fit_args = {
    "x": x_train,
    "y": y_train,
    "epochs": 200,
    "validation_data": (x_val, y_val),
    "batch_size": batch_size,
    'callbacks': callbacks,
    "p_id_col": dm.PROFILE_ID_COL,
    "window_size": window_size,
}

tuner.search(**fit_args)

# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()
