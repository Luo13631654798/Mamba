import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
from gratif import gratif_layers
from torch.nn.utils.rnn import pad_sequence
import pdb

class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.

    # x_delta_t: Tensor # B×K×D: the input intervals.
    # y_delta_t: Tensor # B×K×D: the target intervals.

class Batch_New(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.

    x_delta_t: Tensor # B×K×D: the input intervals.
    y_delta_t: Tensor # B×K×D: the target intervals.

class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor

class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor



def tsdm_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    context_x: list[Tensor] = []
    context_vals: list[Tensor] = []
    context_mask: list[Tensor] = []
    target_vals: list[Tensor] = []
    target_mask: list[Tensor] = []


    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get whole time interval
        sorted_idx = torch.argsort(t)

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()

        # nan to zeros
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)


        # x_vals.append(x[sorted_idx])
        # x_time.append(t[sorted_idx])
        # x_mask.append(mask_x[sorted_idx])
        #
        # y_time.append(t_target)
        # y_vals.append(y)
        # y_mask.append(mask_y)
        
        context_x.append(torch.cat([t, t_target], dim = 0))
        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))
        # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        x_time=pad_sequence(context_x, batch_first=True).squeeze(),
        x_vals=pad_sequence(context_vals, batch_first=True, padding_value=0).squeeze(),
        x_mask=pad_sequence(context_mask, batch_first=True).squeeze(),
        y_time=pad_sequence(context_x, batch_first=True).squeeze(),
        y_vals=pad_sequence(target_vals, batch_first=True, padding_value=0).squeeze(),
        y_mask=pad_sequence(target_mask, batch_first=True).squeeze(),
    )


def tsdm_collate_2(batch: list[Sample]) -> Batch_New:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()

        # nan to zeros
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)

        x_vals.append(x)
        x_time.append(t)
        x_mask.append(mask_x)

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)

    x_time_tensor = pad_sequence(x_time, batch_first=True)
    x_vals_tensor = pad_sequence(x_vals, batch_first=True, padding_value=0)
    x_mask_tensor = pad_sequence(x_mask, batch_first=True)
    y_time_tensor = pad_sequence(y_time, batch_first=True)
    y_vals_tensor = pad_sequence(y_vals, batch_first=True, padding_value=0)
    y_mask_tensor = pad_sequence(y_mask, batch_first=True)

    last_obs_tp = torch.full((x_vals_tensor.shape[0], x_vals_tensor.shape[-1]), -1).to(torch.float32)
    x_delta_t = torch.zeros_like(x_vals_tensor)
    y_delta_t = torch.zeros_like(y_vals_tensor)

    for t in range(x_time_tensor.shape[-1]):
        cur_obs_var = torch.where(x_mask_tensor[:, t, :] == 1)
        first_obs_var = torch.where(last_obs_tp[cur_obs_var] == -1)
        x_delta_t[cur_obs_var[0][first_obs_var], t, cur_obs_var[1][first_obs_var]] = 0.02
        has_last_obs_var = torch.where(last_obs_tp[cur_obs_var] != -1)
        x_delta_t[cur_obs_var[0][has_last_obs_var], t, cur_obs_var[1][has_last_obs_var]] = x_time_tensor[cur_obs_var[0][has_last_obs_var], t] - last_obs_tp[cur_obs_var[0][has_last_obs_var], cur_obs_var[1][has_last_obs_var]]
        last_obs_tp[cur_obs_var] = x_time_tensor[cur_obs_var[0], t]

    for t in range(y_time_tensor.shape[-1]):
        cur_obs_var = torch.where(y_mask_tensor[:, t, :] == 1)
        first_obs_var = torch.where(last_obs_tp[cur_obs_var] == -1)
        y_delta_t[cur_obs_var[0][first_obs_var], t, cur_obs_var[1][first_obs_var]] = y_time_tensor[cur_obs_var[0][first_obs_var], t]
        has_last_obs_var = torch.where(last_obs_tp[cur_obs_var] != -1)
        y_delta_t[cur_obs_var[0][has_last_obs_var], t, cur_obs_var[1][has_last_obs_var]] = y_time_tensor[cur_obs_var[0][has_last_obs_var], t] - last_obs_tp[cur_obs_var[0][has_last_obs_var], cur_obs_var[1][has_last_obs_var]]
        last_obs_tp[cur_obs_var] = y_time_tensor[cur_obs_var[0], t]


    return Batch_New(
        x_time=x_time_tensor,
        x_vals=x_vals_tensor,
        x_mask=x_mask_tensor,
        y_time=y_time_tensor,
        y_vals=y_vals_tensor,
        y_mask=y_mask_tensor,
        x_delta_t=x_delta_t,
        y_delta_t=y_delta_t
    )


class GrATiF(nn.Module):

    def __init__(
        self,
        input_dim=41,
        attn_head=4,
        latent_dim = 128,
        n_layers=2,
        device='cuda'):
        super().__init__()
        self.dim=input_dim
        self.attn_head=attn_head
        self.latent_dim = latent_dim
        self.n_layers=n_layers
        self.device=device
        self.enc = gratif_layers.Encoder(self.dim, self.latent_dim, self.n_layers, self.attn_head, device=device)

    def get_extrapolation(self, context_x, context_w, target_x, target_y):
        context_mask = context_w[:, :, self.dim:]
        X = context_w[:, :, :self.dim]
        X = X*context_mask
        context_mask = context_mask + target_y[:,:,self.dim:]
        output, target_U_, target_mask_ = self.enc(context_x, X, context_mask, target_y[:,:,:self.dim], target_y[:,:,self.dim:])
        return output, target_U_, target_mask_

    def convert_data(self,  x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        return x_time, torch.cat([x_vals, x_mask],-1), y_time, torch.cat([y_vals, y_mask],-1)  

    def forward(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        context_x, context_y, target_x, target_y = self.convert_data(x_time, x_vals, x_mask, y_time, y_vals, y_mask)
        # pdb.set_trace()
        if len(context_y.shape) == 2:
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
            target_x = target_x.unsqueeze(0)
            target_y = target_y.unsqueeze(0)
        output, target_U_, target_mask_ = self.get_extrapolation(context_x, context_y, target_x, target_y)

        return output, target_U_, target_mask_.to(torch.bool)