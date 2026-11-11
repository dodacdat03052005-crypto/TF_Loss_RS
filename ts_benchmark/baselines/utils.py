# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ts_benchmark.baselines.time_series_library.utils.timefeatures import (
    time_features,
)
from ts_benchmark.utils.data_processing import split_time

class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """

    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha

    def forward(self, x, alpha_override=None):
        _, t, _ = x.shape
        a = self.alpha if alpha_override is None else float(alpha_override)
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - a), powers).to(x.device)
        divisor = weights.clone()
        weights[1:] = weights[1:] * a
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)


class DECOMP(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, alpha):
        super(DECOMP, self).__init__()
        self.ma = EMA(alpha)

    def forward(self, x, alpha_override=None):
        moving_average = self.ma(x, alpha_override=alpha_override)
        res = x - moving_average
        return res, moving_average


class DBLoss(nn.Module):
    """自定义分解损失函数（趋势+季节双损失）"""

    def __init__(self, alpha, beta):
        super().__init__()
        self.decomp = DECOMP(alpha)
        self.beta = beta
        self.mse = nn.MSELoss(reduction="mean")
        self.mae = nn.L1Loss(reduction="mean")

    def forward(self, pred, target):
        pred_season, pred_trend = self.decomp(pred)
        target_season, target_trend = self.decomp(target)

        season_loss = self.mse(pred_season, target_season)
        trend_loss = self.mae(pred_trend, target_trend)
        trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()
        return self.beta * season_loss + (1 - self.beta) * trend_loss


class HD_DBLoss(nn.Module):
    """
    Horizon-Adaptive Heteroscedastic Decomposition-Based Loss (HD-DBLoss)
    Adapts decomposition smoothing, branch balance, and heteroscedastic decay to horizon H:
      1. gamma(H):
         - 'mixture' (Two-Regime Horizon Mixture Law, default):
             gamma(H) = gamma - kappa_gamma * (1 - H_ref / H)
           Derives from the exact proportion of short-term steps (H_ref / H) vs
           long-term steps (1 - H_ref / H) in a forecast window of length H.
         - 'log':
             gamma(H) = gamma - kappa_gamma * ln(H / H_ref)
      2. alpha(H) = alpha * (H_ref / H) ** eta_alpha
      3. beta(H) = clip(beta + kappa_beta * ln(H / H_ref), 0.05, 0.95)
    """

    def __init__(
        self,
        alpha=0.2,
        beta=0.5,
        gamma=1.15,
        gamma_season=1.15,
        tau=4.0,
        h_ref=96.0,
        kappa_gamma=1.75,
        eta_alpha=0.35,
        kappa_beta=0.25,
        gamma_mode="mixture",
        val_unweighted=False,
        h0=0.0,
        trend_loss_type="mae",
        align_scale=True,
        learnable_gamma=False,
        lambda_reg=0.1,
        lambda_trend=1.0,
        lambda_season=1.0,
    ):
        super().__init__()
        self.decomp = DECOMP(alpha)
        self.alpha_init = float(alpha)
        self.beta_init = float(beta)
        self.gamma_init = float(gamma)
        self.gamma_season_init = float(gamma_season)
        self.tau = float(tau)
        self.h_ref = float(h_ref)
        self.kappa_gamma = float(kappa_gamma)
        self.eta_alpha = float(eta_alpha)
        self.kappa_beta = float(kappa_beta)
        self.gamma_mode = str(gamma_mode).lower()
        self.val_unweighted = bool(val_unweighted)
        self.h0 = float(h0)
        self.trend_loss_type = str(trend_loss_type).lower()
        self.align_scale = bool(align_scale)
        self.learnable_gamma = bool(learnable_gamma)
        self.lambda_reg = float(lambda_reg)
        self.lambda_trend = float(lambda_trend)
        self.lambda_season = float(lambda_season)

        if self.learnable_gamma:
            self.gamma_raw = nn.Parameter(torch.tensor(float(gamma), dtype=torch.float32))
        else:
            self.register_parameter("gamma_raw", None)

    def _horizon_params(self, H_steps: int):
        """Compute horizon-dependent (alpha_H, beta_H, gamma_T_H, gamma_S_H) from H_steps."""
        ratio = max(float(H_steps) / max(self.h_ref, 1.0), 1e-4)
        log_r = float(np.log(ratio))
        mix_r = 1.0 - (1.0 / ratio)  # 1 - H_ref / H: fraction of long-term steps in [0, 1)

        # 1. EMA smoothing alpha(H): wider smoothing window for longer horizons
        alpha_H = self.alpha_init * (ratio ** (-self.eta_alpha))
        alpha_H = float(np.clip(alpha_H, 0.02, 0.95))

        # 2. Branch balance beta(H)
        beta_H = float(np.clip(self.beta_init + self.kappa_beta * log_r, 0.05, 0.95))

        # 3. Heteroscedastic decay exponent gamma(H)
        r_gamma = mix_r if self.gamma_mode == "mixture" else log_r
        gamma_T_H = self.gamma_init - self.kappa_gamma * r_gamma
        gamma_S_H = self.gamma_season_init - self.kappa_gamma * r_gamma
        return alpha_H, beta_H, gamma_T_H, gamma_S_H

    def forward(self, pred, target):
        device = pred.device
        dtype = pred.dtype
        _, H_steps, _ = pred.shape

        alpha_H, beta_H, gamma_T_H, gamma_S_H = self._horizon_params(H_steps)

        # 1. Horizon-adapted EMA Decomposition
        pred_season, pred_trend = self.decomp(pred, alpha_override=alpha_H)
        target_season, target_trend = self.decomp(target, alpha_override=alpha_H)

        # 2. Pointwise Seasonal (MSE) and Trend (MAE/MSE) errors [Batch, H, Channels]
        season_err = (pred_season - target_season) ** 2
        if self.trend_loss_type == "mse":
            trend_err = (pred_trend - target_trend) ** 2
        else:
            trend_err = torch.abs(pred_trend - target_trend)

        # Validation / EarlyStopping evaluation alignment (fixes DBLoss validation blindspot)
        if self.val_unweighted and not torch.is_grad_enabled():
            mse_val = torch.mean((pred - target) ** 2)
            mae_val = torch.mean(torch.abs(pred - target))
            return 0.5 * mse_val + 0.5 * mae_val

        # 3. Normalized heteroscedastic weights w_h(H) across horizon [1, H, 1]
        h = torch.arange(1, H_steps + 1, device=device, dtype=dtype).view(1, H_steps, 1)
        denom = max(H_steps - 1, 1)
        s_h = 1.0 + self.tau * ((h - 1.0) / denom) if self.tau > 0 else (h + self.h0)

        if abs(gamma_S_H) > 1e-6:
            w_s_tilde = torch.pow(s_h, -gamma_S_H)
            w_s = w_s_tilde / (torch.mean(w_s_tilde) + 1e-8)
            season_loss = torch.mean(season_err * w_s)
        else:
            season_loss = torch.mean(season_err)

        gamma_val = self.gamma_raw if self.learnable_gamma else gamma_T_H
        if self.learnable_gamma or abs(gamma_T_H) > 1e-6:
            w_tilde = torch.pow(s_h, -gamma_val)
            w = w_tilde / (torch.mean(w_tilde) + 1e-8)
            trend_loss = torch.mean(trend_err * w)
        else:
            trend_loss = torch.mean(trend_err)

        # 4. Dynamic scale alignment (identical to DBLoss)
        if self.align_scale:
            trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()
            return beta_H * season_loss + (1.0 - beta_H) * trend_loss

        return self.lambda_trend * trend_loss + self.lambda_season * season_loss


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.lr}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        improved = False
        if self.best_score is None:
            self.best_score = score
            improved = True
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            improved = True
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
            self.val_loss_min = val_loss
            self.counter = 0
        return improved


class SlidingWindowDataLoader:
    """
    SlidingWindDataLoader class.

    This class encapsulates a sliding window data loader for generating time series training samples.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_size: int = 1,
        history_length: int = 10,
        prediction_length: int = 2,
        shuffle: bool = True,
    ):
        """
        Initialize SlidingWindDataLoader.

        :param dataset: Pandas DataFrame containing time series data.
        :param batch_size: Batch size.
        :param history_length: The length of historical data.
        :param prediction_length: The length of the predicted data.
        :param shuffle: Whether to shuffle the dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.shuffle = shuffle
        self.current_index = 0

    def __len__(self) -> int:
        """
        Returns the length of the data loader.

        :return: The length of the data loader.
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __iter__(self) -> "SlidingWindowDataLoader":
        """
        Create an iterator and return.

        :return: Data loader iterator.
        """
        if self.shuffle:
            self._shuffle_dataset()
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data for the next batch.

        :return: A tuple containing input data and target data.
        """
        if self.current_index >= len(self):
            raise StopIteration

        batch_inputs = []
        batch_targets = []
        for _ in range(self.batch_size):
            window_data = self.dataset.iloc[
                self.current_index : self.current_index
                + self.history_length
                + self.prediction_length,
                :,
            ]
            if len(window_data) < self.history_length + self.prediction_length:
                raise StopIteration  # Stop iteration when the dataset is less than one window size and prediction step size

            inputs = window_data.iloc[: self.history_length].values
            targets = window_data.iloc[
                self.history_length : self.history_length + self.prediction_length
            ].values

            batch_inputs.append(inputs)
            batch_targets.append(targets)
            self.current_index += 1

        # Convert NumPy array to PyTorch tensor
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32)

        return batch_inputs, batch_targets

    def _shuffle_dataset(self):
        """
        Shuffle the dataset.
        """
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)


def train_val_split(train_data, ratio, seq_len):
    if ratio == 1:
        return train_data, None

    elif seq_len is not None:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_time(train_data, border)
        train_data_rest, valid_data = split_time(train_data, border - seq_len)
        return train_data_value, valid_data
    else:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_time(train_data, border)
        return train_data_value, valid_data_rest


def decompose_time(
    time: np.ndarray,
    freq: str,
) -> np.ndarray:
    """
    Split the given array of timestamps into components based on the frequency.

    :param time: Array of timestamps.
    :param freq: The frequency of the time stamp.
    :return: Array of timestamp components.
    """
    df_stamp = pd.DataFrame(pd.to_datetime(time), columns=["date"])
    freq_scores = {
        "m": 0,
        "w": 1,
        "b": 2,
        "d": 2,
        "h": 3,
        "t": 4,
        "s": 5,
    }
    max_score = max(freq_scores.values())
    df_stamp["month"] = df_stamp.date.dt.month
    if freq_scores.get(freq, max_score) >= 1:
        df_stamp["day"] = df_stamp.date.dt.day
    if freq_scores.get(freq, max_score) >= 2:
        df_stamp["weekday"] = df_stamp.date.dt.weekday
    if freq_scores.get(freq, max_score) >= 3:
        df_stamp["hour"] = df_stamp.date.dt.hour
    if freq_scores.get(freq, max_score) >= 4:
        df_stamp["minute"] = df_stamp.date.dt.minute
    if freq_scores.get(freq, max_score) >= 5:
        df_stamp["second"] = df_stamp.date.dt.second
    return df_stamp.drop(["date"], axis=1).values


def get_time_mark(
    time_stamp: np.ndarray,
    timeenc: int,
    freq: str,
) -> np.ndarray:
    """
    Extract temporal features from the time stamp.

    :param time_stamp: The time stamp ndarray.
    :param timeenc: The time encoding type.
    :param freq: The frequency of the time stamp.
    :return: The mark of the time stamp.
    """
    if timeenc == 0:
        origin_size = time_stamp.shape
        data_stamp = decompose_time(time_stamp.flatten(), freq)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 1:
        origin_size = time_stamp.shape
        data_stamp = time_features(pd.to_datetime(time_stamp.flatten()), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    else:
        raise ValueError("Unknown time encoding {}".format(timeenc))
    return data_stamp.astype(np.float32)


def forecasting_data_provider(data, config, timeenc, batch_size, shuffle, drop_last):
    dataset = DatasetForTransformer(
        dataset=data,
        history_len=config.seq_len,
        prediction_len=config.pred_len,
        label_len=config.label_len,
        timeenc=timeenc,
        freq=config.freq,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )

    return dataset, data_loader


class DatasetForTransformer:
    def __init__(
        self,
        dataset: pd.DataFrame,
        history_len: int = 10,
        prediction_len: int = 2,
        label_len: int = 5,
        timeenc: int = 1,
        freq: str = "h",
    ):
        # init

        self.dataset = dataset
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.label_length = label_len
        self.current_index = 0
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()

    def __len__(self) -> int:
        """
        Returns the length of the data loader.

        :return: The length of the data loader.
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __read_data__(self):
        df_stamp = self.dataset.reset_index()
        df_stamp = df_stamp[["date"]].values.transpose(1, 0)
        data_stamp = get_time_mark(df_stamp, self.timeenc, self.freq)[0]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_length
        r_begin = s_end - self.label_length
        r_end = r_begin + self.label_length + self.prediction_length

        seq_x = self.dataset[s_begin:s_end]
        seq_y = self.dataset[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x.values, dtype=torch.float32)
        seq_y = torch.tensor(seq_y.values, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark


class SegLoader(object):
    def __init__(self, data, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data = data
        self.test_labels = data

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.data[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "val":
            return np.float32(self.data[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.data[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.data[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


def anomaly_detection_data_provider(
    data, batch_size, win_size=100, step=100, mode="train"
):
    dataset = SegLoader(data, win_size, 1, mode)

    shuffle = False
    if mode == "train" or mode == "val":
        shuffle = True

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
    return data_loader
