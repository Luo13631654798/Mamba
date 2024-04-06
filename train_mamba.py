import argparse
import sys
import time
from random import SystemRandom
from mamba import Net, Mamba_Seq2Seq
# fmt: off
parser = argparse.ArgumentParser(description="Training Script for USHCN dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=300,    type=int,   help="maximum epochs")
parser.add_argument("-f",  "--fold",         default=2,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=128,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-hd", "--hidden-dim",  default=64,    type=int,   help="hidden-dim")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-s",  "--seed",         default=None,   type=int,   help="Set the random seed.")
parser.add_argument("-nl",  "--nlayers", default=2,   type=int,   help="")
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-ft", "--forc-time", default=6, type=int, help="forecast horizon in hours")
parser.add_argument("-ct", "--cond-time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="# folds for crossvalidation")
parser.add_argument("--cuda", default=0, type=int)

import pdb
# fmt: on

ARGS = parser.parse_args()
print(' '.join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)
from mamba_origin import Mamba
import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)

import logging
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import torchinfo
from IPython.core.display import HTML
from torch import Tensor, jit
import pdb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from torch.optim import AdamW
from gratif.gratif import tsdm_collate, tsdm_collate_2
warnings.filterwarnings(action="ignore", category=UserWarning)
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")

def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.mean((y[mask] - yhat[mask])**2)
    return err

def MAE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sum(mask*torch.abs(y-yhat), 1)/(torch.sum(mask,1))
    return torch.mean(err)

def RMSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sqrt(torch.sum(mask*(y-yhat)**2, 1)/(torch.sum(mask,1)))
    return torch.mean(err)

def predict_fn(model, batch) -> tuple[Tensor, Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, TY, Y, MY, DTX, DTY = (tensor.to(DEVICE) for tensor in batch)
    output, target_U_, target_mask_ = model(T, X, M, TY, Y, MY, DTX, DTY)
    return target_U_, output.squeeze(), target_mask_

def predict_fn_inference(model, batch) -> tuple[Tensor, Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, TY, Y, MY, DTX, DTY = (tensor.to(DEVICE) for tensor in batch)
    output, target_U_, target_mask_ = model.inference(T, X, M, TY, Y, MY, DTX, DTY)
    return target_U_, output.squeeze(), target_mask_

def predict_fn_2(model, batch, max_cond_len, max_forc_len) -> tuple[Tensor, Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, TY, Y, MY, DTX, DTY = (tensor.to(DEVICE) for tensor in batch)
    batch_size, cur_x_len, var_nums = X.shape
    batch_size, cur_y_len, var_nums = Y.shape
    T_extend = torch.zeros(size=[batch_size, max_cond_len]).to(DEVICE)
    T_extend[:, :cur_x_len] = T

    X_extend = torch.zeros(size=[batch_size, max_cond_len, var_nums]).to(DEVICE)
    X_extend[:, :cur_x_len, :] = X

    M_extend = torch.zeros(size=[batch_size, max_cond_len, var_nums]).to(DEVICE)
    M_extend[:, :cur_x_len, :] = M

    DTX_extend = torch.zeros(size=[batch_size, max_cond_len, var_nums]).to(DEVICE)
    DTX_extend[:, :cur_x_len, :] = DTX

    TY_extend = torch.zeros(size=[batch_size, max_forc_len]).to(DEVICE)
    TY_extend[:, :cur_y_len] = TY

    Y_extend = torch.zeros(size=[batch_size, max_forc_len, var_nums]).to(DEVICE)
    Y_extend[:, :cur_y_len, :] = Y

    MY_extend = torch.zeros(size=[batch_size, max_forc_len, var_nums]).to(DEVICE)
    MY_extend[:, :cur_y_len, :] = MY

    DTY_extend = torch.zeros(size=[batch_size, max_forc_len, var_nums]).to(DEVICE)
    DTY_extend[:, :cur_y_len, :] = DTY


    output, target_U_, target_mask_ = model(T_extend, X_extend, M_extend, TY_extend, Y_extend, MY_extend, DTX_extend, DTY_extend)
    return target_U_, output.squeeze(), target_mask_


OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

if ARGS.dataset=="ushcn":
    from tsdm.tasks import USHCN_DeBrouwer2019
    TASK = USHCN_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
    max_cond_len = 288
    max_forc_len = 3
elif ARGS.dataset=="mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
    TASK = MIMIC_III_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
    max_cond_len = 2 * ARGS.cond_time + 1
    max_forc_len = 2 * ARGS.forc_time if ARGS.forc_time !=0 else 3
elif ARGS.dataset=="mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
    TASK = MIMIC_IV_Bilos2021(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
    max_cond_len = 482 if ARGS.cond_time == 24 else 707
    if ARGS.forc_time == 0:
        max_forc_len = 3
    else:
        max_forc_len = 488 if ARGS.cond_time == 24 else 707
elif ARGS.dataset=='physionet2012':
    from tsdm.tasks.physionet2012 import Physionet2012
    TASK = Physionet2012(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon = ARGS.forc_time, num_folds=ARGS.nfolds)
    if ARGS.forc_time == 0:
        max_cond_len = 37
        max_forc_len = 3
    elif ARGS.cond_time == 24 and ARGS.forc_time == 12:
        max_cond_len = 25
        max_forc_len = 12
    elif ARGS.cond_time == 24 and ARGS.forc_time == 24:
        max_cond_len = 25
        max_forc_len = 23
    elif ARGS.cond_time == 36 and ARGS.forc_time == 6:
        max_cond_len = 36
        max_forc_len = 6
    elif ARGS.cond_time == 36 and ARGS.forc_time == 12:
        max_cond_len = 37
        max_forc_len = 11



DEVICE = torch.device('cuda:' + str(ARGS.cuda) if torch.cuda.is_available() else 'cpu')

METRICS = {
    "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    "MAE": jit.script(MAE),
}

LOSS = jit.script(MSE)

dloader_config_train = {
    "batch_size": ARGS.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": tsdm_collate_2,
}

dloader_config_infer = {
    "batch_size": 64,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": tsdm_collate_2,
}

test_loss_list = []


for fold in range(ARGS.nfolds):
    print("------Fold " + str(fold) + " BEGINNING------")
    # Set different random seed
    torch.manual_seed(fold)
    torch.cuda.manual_seed(fold)
    np.random.seed(fold)
    random.seed(fold)

    TRAIN_LOADER = TASK.get_dataloader((fold, "train"), **dloader_config_train)
    # INFER_LOADER = TASK.get_dataloader((fold, "train"), **dloader_config_infer)
    VALID_LOADER = TASK.get_dataloader((fold, "valid"), **dloader_config_infer)
    TEST_LOADER = TASK.get_dataloader((fold, "test"), **dloader_config_infer)
    EVAL_LOADERS = {"train": TRAIN_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}

    # max_cond_len = 0
    # max_forc_len = 0
    # for batch in (TRAIN_LOADER):
    #     max_cond_len = batch.x_time.shape[-1] if batch.x_time.shape[-1] > max_cond_len else max_cond_len
    #     max_forc_len = batch.y_time.shape[-1] if batch.y_time.shape[-1] > max_forc_len else max_forc_len
    #
    # for batch in (INFER_LOADER):
    #     max_cond_len = batch.x_time.shape[-1] if batch.x_time.shape[-1] > max_cond_len else max_cond_len
    #     max_forc_len = batch.y_time.shape[-1] if batch.y_time.shape[-1] > max_forc_len else max_forc_len
    #
    # for batch in (VALID_LOADER):
    #     max_cond_len = batch.x_time.shape[-1] if batch.x_time.shape[-1] > max_cond_len else max_cond_len
    #     max_forc_len = batch.y_time.shape[-1] if batch.y_time.shape[-1] > max_forc_len else max_forc_len
    #
    # for batch in (TEST_LOADER):
    #     max_cond_len = batch.x_time.shape[-1] if batch.x_time.shape[-1] > max_cond_len else max_cond_len
    #     max_forc_len = batch.y_time.shape[-1] if batch.y_time.shape[-1] > max_forc_len else max_forc_len

    var_nums = TASK.dataset.shape[-1]
    # MODEL_CONFIG = {
    #         "input_dim":TASK.dataset.shape[-1],
    #         "attn_head":ARGS.attn_head,
    #         "latent_dim" : ARGS.latent_dim,
    #         "n_layers":ARGS.nlayers,
    #         "device": DEVICE
    # }

    # MODEL = GrATiF(**MODEL_CONFIG).to(DEVICE)
    # This module uses roughly 3 * expand * d_model^2 parameters
    # MODEL = Mamba(
    #     lookback_window = ARGS.cond_time,
    #     forecast_window = ARGS.forc_time,
    #     d_model=TASK.dataset.shape[-1],  # Model dimension d_model
    #     d_state=TASK.dataset.shape[-1],  # SSM state expansion factor
    #     d_conv=4,  # Local convolution width
    #     expand=1,  # Block expansion factor
    #     bias=True
    # ).to(DEVICE)

    MODEL = Net(
        # in_dim=var_nums,
        in_dim=2 * var_nums + 1,
        out_dim=var_nums,
        hidden_dim=ARGS.hidden_dim,
        n_layers=ARGS.nlayers,
        cond_len=max_cond_len,
        forc_len=max_forc_len
    ).to(DEVICE)

    # MODEL = Mamba_Seq2Seq(
    #     in_dim=2 * var_nums + 1,
    #     out_dim=var_nums,
    #     hidden_dim=ARGS.hidden_dim,
    #     n_layers=ARGS.nlayers,
    # ).to(DEVICE)


    torchinfo.summary(MODEL)

    OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, 'min', patience=10, factor=0.5, min_lr=0.00001, verbose=True)

    es = False
    best_val_loss = 10e8
    total_num_batches = 0

    for epoch in range(1, ARGS.epochs+1):
        loss_list = []
        start_time = time.time()
        for batch in (TRAIN_LOADER):
            total_num_batches += 1
            OPTIMIZER.zero_grad()
            # Y, YHAT, MASK = predict_fn(MODEL, batch)
            Y, YHAT, MASK = predict_fn_2(MODEL, batch, max_cond_len, max_forc_len)
            R = LOSS(Y, YHAT, MASK)
            assert torch.isfinite(R).item(), "Model Collapsed!"
            loss_list.append([R])
            # Backward
            R.backward()
            OPTIMIZER.step()
        epoch_time = time.time()
        train_loss = torch.mean(torch.Tensor(loss_list))
        loss_list = []
        count = 0
        with torch.no_grad():
            for batch in (VALID_LOADER):
                total_num_batches += 1
                # Forward
                # Y, YHAT, MASK = predict_fn(MODEL, batch)
                # Y, YHAT, MASK = predict_fn_inference(MODEL, batch)
                Y, YHAT, MASK = predict_fn_2(MODEL, batch, max_cond_len, max_forc_len)
                R = LOSS(Y, YHAT, MASK)
                if R.isnan():
                    pdb.set_trace()
                loss_list.append([R*MASK.sum()])
                count += MASK.sum()
        val_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE)/count)
        print(epoch,"Train: ", train_loss.item(), " VAL: ",val_loss.item(), " epoch time: ", int(epoch_time - start_time), 'secs')
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({    'args': ARGS,
                            'epoch': epoch,
                            'state_dict': MODEL.state_dict(),
                            'optimizer_state_dict': OPTIMIZER.state_dict(),
                            'loss': train_loss,
                        }, 'saved_models/'+ARGS.dataset + '_' + str(experiment_id) + '.h5')
            early_stop = 0
        else:
            early_stop += 1
        if early_stop == 30:
            print("Early stopping because of no improvement in val. metric for 30 epochs")
            es = True
        scheduler.step(val_loss)


        # LOGGER.log_epoch_end(epoch)
        if (epoch == ARGS.epochs) or (es == True):
            chp = torch.load('saved_models/' + ARGS.dataset + '_' + str(experiment_id) + '.h5')
            MODEL.load_state_dict(chp['state_dict'])
            loss_list = []
            count = 0
            with torch.no_grad():
                for batch in (TEST_LOADER):

                    total_num_batches += 1
                    # Forward
                    # Y, YHAT, MASK = predict_fn_inference(MODEL, batch)
                    Y, YHAT, MASK = predict_fn_2(MODEL, batch, max_cond_len, max_forc_len)
                    R = LOSS(Y, YHAT, MASK)
                    assert torch.isfinite(R).item(), "Model Collapsed!"
                    # loss_list.append([R*Y.shape[0]])
                    loss_list.append([R*MASK.sum()])
                    count += MASK.sum()
            test_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE)/count)
            print("Fold " + str(fold) + "------Best_val_loss: ",best_val_loss.item(), " test_loss : ", test_loss.item())
            test_loss_list.append(test_loss.item())
            break
print("Final: test_loss:" + str(np.mean(test_loss_list)) + " ± " + str(np.std(test_loss_list)))