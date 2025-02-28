from torch.nn.utils.rnn import pad_sequence
from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)
    

def compute_losses_of_variable_length_durations(pred_dur, gt_dur, input_lengths):
    # Assume these are your lists:
    # pred_dur: list of tensors, each of shape [L_i, max_dur]
    # gt_dur:   list of tensors, each of shape [L_i] (groundtruth duration for each text frame)
    # input_lengths: list of ints (each L_i)

    device = pred_dur[0].device  # assume all tensors are on the same device

    # Truncate each tensor to its valid length.
    pred_truncated = [pred[:l] for pred, l in zip(pred_dur, input_lengths)]
    gt_truncated = [gt[:l].long() for gt, l in zip(gt_dur, input_lengths)]

    # Pad sequences along the time dimension (batch_first=True)
    # pred_padded: [B, L_max, max_dur]
    pred_padded = pad_sequence(pred_truncated, batch_first=True)
    # gt_padded: [B, L_max]
    gt_padded = pad_sequence(gt_truncated, batch_first=True)

    B, L_max, max_dur = pred_padded.shape

    # ---------------------------
    # Create the target tensor s2s_trg in parallel.
    # For each sample and each valid time frame p, we want to set:
    #    s2s_trg[p, :gt_padded[p]] = 1
    # We do this by comparing a column index range against gt_padded.
    col_range = torch.arange(max_dur, device=device).view(
        1, 1, max_dur)  # shape: [1, 1, max_dur]
    s2s_trg = (col_range < gt_padded.unsqueeze(
        2)).float()  # shape: [B, L_max, max_dur]

    # ---------------------------
    # Compute the predicted duration per time frame.
    # Apply sigmoid to pred_padded and sum over the max_dur dimension.
    dur_pred = torch.sigmoid(pred_padded).sum(dim=2)  # shape: [B, L_max]
    # mask the elements of dur_pred exceeding the input_lengths
    col_range = torch.arange(L_max, device=device).view(1, L_max)
    lengths_tensor = torch.tensor(input_lengths, device=device).view(B, 1)
    samplewise_length_valid_mask = (col_range < lengths_tensor).float()
    dur_pred = dur_pred * samplewise_length_valid_mask # length of each sample

    # ---------------------------
    # Create masks for valid time steps.
    # We'll need these to compute the losses only on valid parts.
    time_range = torch.arange(
        L_max, device=device).unsqueeze(0)   # shape: [1, L_max]

    # For the L1 loss, exclude the first and last valid time step (similar to [1:_text_length-1])
    inner_mask = (time_range >= 1) & (
        time_range < (lengths_tensor.unsqueeze(1) - 1)).squeeze()
    inner_mask = inner_mask.unsqueeze(2)  # shape: [B, L_max, 1]
    framewise_duration_valid_mask = (s2s_trg * samplewise_length_valid_mask.unsqueeze(-1)).bool()

    # ---------------------------
    # Compute the losses in parallel.
    # L1 loss on duration prediction (only inner valid time steps)
    # dur_pred shape : [B, L_max], gt_padded shape : [B, L_max]
    loss_dur = F.l1_loss(dur_pred[inner_mask.squeeze()], gt_padded[inner_mask.squeeze()].float(), reduction='sum')
    loss_dur = loss_dur / torch.sum(inner_mask)

    # Binary Cross-Entropy loss with logits on the entire valid region.
    # We flatten only the valid time steps from each sample.
    # shape: [num_valid, max_dur]
    # pred_valid = pred_padded[elementwise_valid_mask].view(-1, max_dur)
    # trg_valid = s2s_trg[elementwise_valid_mask].view(-1, max_dur)
    pred_valid = pred_padded[framewise_duration_valid_mask.bool()]
    trg_valid = s2s_trg[framewise_duration_valid_mask.bool()]
    loss_ce = F.binary_cross_entropy_with_logits(pred_valid, trg_valid, reduction="sum")
    loss_ce = loss_ce / torch.sum(framewise_duration_valid_mask)

    return loss_dur, loss_ce