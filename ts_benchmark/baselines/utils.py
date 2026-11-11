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

    def forward(self, x):
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to(x.device)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
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

    def forward(self, x):
        moving_average = self.ma(x)
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
    Heteroscedastic Decomposition-Based Loss (HD-DBLoss)
    - Trend branch (Non-stationary I(1)):
      Weighted by power-law inverse variance w_T(h) ~ s_h^(-gamma) + First-Difference velocity loss |dT_h - dT_hat_h|.
    - Seasonal branch (Stationary I(0), Wold's Decomposition):
      Conditional h-step forecast variance grows and saturates with h -> weighted by w_S(h) ~ s_h^(-gamma_season)
      with optional Huber/Charbonnier outlier damping at far horizons to prevent L2 gradient explosion.
    - Dynamic scale alignment matching DBLoss magnitude balancing.
    """

    def __init__(
        self,
        alpha=0.2,
        beta=0.5,
        gamma=0.5,
        gamma_season=0.5,
        gamma_min=0.1,
        tau=3.0,
        h0=0.0,
        delta0=0.02,
        season_delta=0.5,
        diff_weight=0.35,
        bi_weight=0.0,
        trend_loss_type="charbonnier",
        align_scale=True,
        gamma_anneal=False,
        eval_uniform=False,
        learnable_gamma=False,
        lambda_reg=0.1,
        lambda_trend=1.0,
        lambda_season=1.0,
    ):
        super().__init__()
        self.decomp = DECOMP(alpha)
        self.beta = float(beta)
        self.gamma_init = float(gamma)
        self.gamma_season = float(gamma_season)
        self.gamma_min = float(gamma_min)
        self.current_gamma = float(gamma)
        self.tau = float(tau)
        self.h0 = float(h0)
        self.delta0 = float(delta0)
        self.season_delta = float(season_delta)
        self.diff_weight = float(diff_weight)
        self.bi_weight = float(bi_weight)
        self.trend_loss_type = str(trend_loss_type).lower()
        self.align_scale = bool(align_scale)
        self.gamma_anneal = bool(gamma_anneal)
        self.eval_uniform = bool(eval_uniform)
        self.learnable_gamma = bool(learnable_gamma)
        self.lambda_reg = float(lambda_reg)
        self.lambda_trend = float(lambda_trend)
        self.lambda_season = float(lambda_season)

        if self.learnable_gamma:
            init_val = max(self.gamma_init, 1e-3)
            raw_init = float(np.log(np.expm1(init_val)))
            self.gamma_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        else:
            self.register_parameter("gamma_raw", None)

    def set_epoch(self, epoch: int, num_epochs: int):
        """Update effective gamma via cosine annealing schedule during training."""
        if self.gamma_anneal and not self.learnable_gamma and num_epochs > 1:
            progress = min(max(epoch / float(num_epochs - 1), 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            self.current_gamma = self.gamma_min + (self.gamma_init - self.gamma_min) * float(cosine)
        else:
            self.current_gamma = self.gamma_init

    def _decompose(self, x):
        season_f, trend_f = self.decomp(x)
        if self.bi_weight > 0.0:
            x_rev = torch.flip(x, dims=[1])
            _, trend_b = self.decomp(x_rev)
            trend_b = torch.flip(trend_b, dims=[1])
            trend = (1.0 - self.bi_weight) * trend_f + self.bi_weight * trend_b
            season = x - trend
            return season, trend
        return season_f, trend_f

    def _compute_trend_error(self, pred_trend, target_trend, s_h, gamma_val):
        diff = pred_trend - target_trend
        if self.trend_loss_type == "charbonnier" and self.delta0 > 0.0:
            delta_h = self.delta0 * torch.pow(s_h, gamma_val)
            return torch.sqrt(diff * diff + delta_h * delta_h) - delta_h
        elif self.trend_loss_type == "mae" or (self.trend_loss_type == "charbonnier" and self.delta0 <= 0.0):
            return torch.abs(diff)
        elif self.trend_loss_type == "huber":
            return torch.nn.functional.smooth_l1_loss(
                pred_trend, target_trend, reduction="none", beta=0.5
            )
        else:  # "mse"
            return diff * diff

    def forward(self, pred, target):
        device = pred.device
        dtype = pred.dtype
        _, H_steps, _ = pred.shape

        # 1. Decompose into seasonal and trend components
        pred_season, pred_trend = self._decompose(pred)
        target_season, target_trend = self._decompose(target)

        # 2. Variance scaling profile s_h across horizon [1, H, 1]
        h = torch.arange(1, H_steps + 1, device=device, dtype=dtype).view(1, H_steps, 1)
        if self.tau > 0:
            denom = max(H_steps - 1, 1)
            s_h = 1.0 + self.tau * ((h - 1.0) / denom)
        else:
            s_h = h + self.h0

        gamma_val = (
            torch.nn.functional.softplus(self.gamma_raw)
            if self.learnable_gamma
            else self.current_gamma
        )

        # 3. Seasonal loss (Wold conditional variance decay + optional Huber outlier damping at far h)
        season_diff = pred_season - target_season
        if self.season_delta > 0.0:
            # Smooth L2 for normal seasonal waves, bounded L1 for extreme unpredictable spikes at far h
            delta_s = self.season_delta * torch.pow(s_h, 0.5 * self.gamma_season)
            season_err = 2.0 * (delta_s * delta_s) * (
                torch.sqrt(1.0 + (season_diff / (delta_s + 1e-8)) ** 2) - 1.0
            )
        else:
            season_err = season_diff * season_diff

        if self.gamma_season > 0.0 and not (self.eval_uniform and not torch.is_grad_enabled()):
            w_s_tilde = torch.pow(s_h, -self.gamma_season)
            w_s = w_s_tilde / (torch.mean(w_s_tilde) + 1e-8)
            season_loss = torch.mean(season_err * w_s)
        else:
            season_loss = torch.mean(season_err)

        # 4. Pointwise heteroscedastic trend error [Batch, H, Channels]
        trend_err = self._compute_trend_error(pred_trend, target_trend, s_h, gamma_val)

        if self.eval_uniform and not torch.is_grad_enabled():
            trend_level_loss = torch.mean(trend_err)
        else:
            w_tilde = torch.pow(s_h, -gamma_val)
            w = w_tilde / (torch.mean(w_tilde) + 1e-8)
            trend_level_loss = torch.mean(trend_err * w)
            if self.learnable_gamma and torch.is_grad_enabled():
                trend_level_loss = trend_level_loss + self.lambda_reg * gamma_val * torch.mean(
                    torch.log(s_h)
                )

        # 5. I(1) -> I(0) First-Difference Trend Velocity Loss (couples step h-1 and h)
        if self.diff_weight > 0.0 and H_steps > 1:
            pred_vel = pred_trend[:, 1:, :] - pred_trend[:, :-1, :]
            target_vel = target_trend[:, 1:, :] - target_trend[:, :-1, :]
            vel_err = torch.abs(pred_vel - target_vel)
            vel_loss = torch.mean(vel_err)
            vel_loss = vel_loss * (trend_level_loss / (vel_loss + 1e-8)).detach()
            trend_loss = (1.0 - self.diff_weight) * trend_level_loss + self.diff_weight * vel_loss
        else:
            trend_loss = trend_level_loss

        # 6. Dynamic scale alignment (matches DBLoss magnitude balancing)
        if self.align_scale:
            trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()
            return self.beta * season_loss + (1.0 - self.beta) * trend_loss

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
