# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
import time
from typing import Iterable
import torch
import utils
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

def train_one_epoch(model: torch.nn.Module=None, criterion: torch.nn.Module=None,
                    data_loader: Iterable=None, optimizer: torch.optim.Optimizer=None,
                    device: torch.device=None, logger = None, epoch: int=None, max_norm: float = 0, num_class: int = 4, dataset_name: str="MOSI"):
    
    model.train()
    model = model.to(device)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    logger.output_print(f'\nStart Epoch {epoch} Train.')

    for id, visual_inputs, text_inputs, audio_inputs, label_target, iou_scores in metric_logger.log_every(data_loader, print_freq, header):
        B, T, D = visual_inputs.shape 
        visual_inputs = visual_inputs.to(device)
        text_inputs = text_inputs.to(device)
        audio_inputs = audio_inputs.to(device)
        label_target = label_target.to(device)
        iou_scores = iou_scores.view(B, 1).to(device)
        
        prob_score, feature_emb = model(visual_inputs, text_inputs, audio_inputs)
    
        outputs = {
            'labels_loss': (prob_score, iou_scores),  
            'contrastive_loss': feature_emb
        }

        targets = {
            'labels_loss': (label_target.view(-1, num_class)), 
            'contrastive_loss':  (label_target.view(-1, num_class))
        }

        loss_dict = criterion(outputs, targets, dataset_name)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        lossue = losses_reduced_scaled.item()

        if not math.isfinite(lossue):
            print("Loss is {}, stopping training".format(lossue))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=lossue, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    logger.output_print(f"Averaged stats: {metric_logger}")

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, pre_micro_f1, criterion, data_loader, device, logger, args, epoch, data_type, results_path = None, nprocs=4):

    model.eval()
    model = model.to(device)
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    all_probs, all_classes, all_ids = [], [], []
    num_class = args.numclass
    best_epoch = False
    logger.output_print(f'\nStart Epoch {epoch} Inference for {data_type} Data.')

    for id, visual_inputs, text_inputs, audio_inputs, label_target, iou_scores in metric_logger.log_every(data_loader, 500, header):

        B, T, D = visual_inputs.shape 
        visual_inputs = visual_inputs.to(device)
        text_inputs = text_inputs.to(device)
        audio_inputs = audio_inputs.to(device)
        label_target = label_target.to(device)
        iou_scores = iou_scores.view(B, 1).to(device)
        id_tensor = id.contiguous().view(B , 2).to(device)
        
        prob_score, feature_emb = model(visual_inputs, text_inputs, audio_inputs)

        outputs = {
            'labels_loss': (prob_score, iou_scores),  
            'contrastive_loss': feature_emb
        }
        
        targets = {
            'labels_loss': (label_target.view(-1, num_class)), 
            'contrastive_loss':  (label_target.view(-1, num_class))
        }

        loss_dict = criterion(outputs, targets, args.dataset_name)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)


        all_ids += list(id_tensor.cpu().numpy())
        all_probs += list(F.softmax(prob_score, dim=-1).cpu().numpy())  
        all_classes += list(label_target.cpu().numpy())

    logger.output_print(f"Averaged stats: {metric_logger}")

    all_ids = np.asarray(all_ids).reshape(-1, num_class).T
    all_probs = np.asarray(all_probs).T
    all_classes = np.asarray(all_classes).T

    results = {'probs': all_probs, 'labels': all_classes}
    map, _, _, _ = utils.frame_level_map_n_cap(results)
    acc, micro_f1 = utils.compute_acc_f1(results)
    logger.output_print('[Epoch-{}] Data Size: {:.4f}'.format(epoch, all_classes.shape[1]))
    logger.output_print('[Epoch-{}] ACC: {:.4f}'.format(epoch, acc))
    logger.output_print('[Epoch-{}] M-F1: {:.4f}'.format(epoch, micro_f1))
    
    if pre_micro_f1 < micro_f1:
        best_epoch = True
        data = np.hstack((all_ids.T, all_probs.T))
        df = pd.DataFrame(data, columns=['video_id', 'cur_time_id', 'prob_pos', 'prob_neg'])
        df[['video_id', 'cur_time_id']] = df[['video_id', 'cur_time_id']].astype(int)
        df.to_csv(results_path, index=False)
        print("\nStore Predict for Best Epoch ", epoch, flush=True)
  
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats, micro_f1, best_epoch
