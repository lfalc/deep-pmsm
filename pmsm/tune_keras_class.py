import keras_tuner
import keras
from preprocessing.cnn_model_utils import CNNKerasRegressor, build_cnn_model
import pandas as pd


class PmsmHyperModel(keras_tuner.HyperModel, CNNKerasRegressor):
    def build(self, hp: keras_tuner.HyperParameters) -> keras.Model:
        n_layers = hp.Int("n_layers", 1, 7)
        n_units = hp.Int("n_units", 2, 200)
        kernel_size = hp.Int("kernel_size", 2, 7)
        regularization_rate = hp.Float(
            "regularization_rate", 1e-9, 1e-1, sampling="log")
        dropout_rate = hp.Float("dropout_rate", 0.2, 0.5)
        lr_rate = hp.Float("lr_rate", 1e-5, 1e1, sampling="log")

        hyper_params = {
            "x_shape": (32, 91),
            "arch": "res",
            "n_layers": n_layers,
            "n_units": n_units,
            "kernel_size": kernel_size,
            "regularization_rate": regularization_rate,
            "dropout_rate": dropout_rate,
            "lr_rate": lr_rate,
        }

        model = build_cnn_model(**hyper_params)
        return model

    def fit(self, hp, model, x, y, callbacks=None, **kwargs):
        assert isinstance(x, pd.DataFrame) and isinstance(
            y, pd.DataFrame
        ), f"{self.__class__.__name__} needs pandas DataFrames as input"

        p_id_col = kwargs.pop("p_id_col", "p_id_col_not_found")
        window_size = kwargs.pop("window_size", None)
        data_cache = kwargs.pop("data_cache", {})
        cache = data_cache.get("data_cache", None)
        batch_size = kwargs.pop("batch_size", None)

        if cache is not None:
            # subsequent conduct iteration
            seq_tra = cache["seq_tra"]
            kwargs["validation_data"] = cache["seq_val"]
        else:
            # first conduct iteration
            seq_tra = self._generate_batches(
                x, y, p_id_col=p_id_col, batch_size=batch_size, window_size=window_size
            )

            x_val, y_val = kwargs.pop("validation_data")
            seq_val = self._generate_batches(
                x_val,
                y_val,
                p_id_col=p_id_col,
                batch_size=batch_size,
                window_size=window_size,
            )

            kwargs["validation_data"] = seq_val
            new_cache = {"seq_tra": seq_tra, "seq_val": seq_val}
            data_cache.update({"data_cache": new_cache})

        fit_args = kwargs
        fit_args['callbacks'] = callbacks
        history = model.fit(seq_tra, **fit_args)

        return history
