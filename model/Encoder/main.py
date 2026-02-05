import argparse
import datetime
import json
import random
import time
from pathlib import Path
from config import get_args_parser
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
import util as utl
import os
import utils
import transformer_models
from dataset import SOICALTASKDataLayer
from train import train_one_epoch, evaluate
import torch.nn as nn
torch.cuda.empty_cache()


def main(args):
    utils.init_distributed_mode(args)
    command = 'python ' + ' '.join(sys.argv)
    this_dir = args.output_dir
    logger = utl.setup_logger(os.path.join(this_dir, 'log_dist.txt'), command=command)
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = transformer_models.VisionTransformer_v3(args=args, inp_dim=args.N, step=args.s,   # VisionTransformer_v3
                                                 patch_dim=args.patch_dim,
                                                 out_dim=args.numclass,
                                                 embedding_dim=args.embedding_dim,
                                                 num_heads=args.num_heads,
                                                 encoder_layers=args.encoder_layers,
                                                 fusion_encoder_layers=args.fusion_encoder_layers,
                                                 hidden_dim=args.hidden_dim,
                                                 dropout_rate=args.dropout_rate,
                                                 attn_dropout_rate=args.attn_dropout_rate,
                                                 num_channels=args.dim_feature,
                                                 positional_encoding_type=args.positional_encoding_type
                                                 )
    
    model.to(device)
    loss_need = [
        'labels_loss',
        'contrastive_loss'
    ]

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.output_print('number of params: {}'.format(n_parameters))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    criterion = utl.SetCriterion(num_classes=args.numclass, losses=loss_need, args=args).to(device)
      
    dataset_val = SOICALTASKDataLayer(phase='test', args=args)
    dataset_train = SOICALTASKDataLayer(phase='train', args=args)

    if not args.eval:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=args.num_workers)
    else:
        data_loader_train = DataLoader(dataset_train, args.batch_size, sampler=sampler_train, drop_last=False, pin_memory=True, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, pin_memory=True, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)
    test_results_path = f'{args.root}/model/Encoder/prediction/{args.dataset_name}/modality_{args.modality}_lenc_{args.encoder_layers}_ldec_{args.fusion_encoder_layers}_alpha_{args.alpha}_beta_{args.beta}_test.csv'
    train_results_path = f'{args.root}/model/Encoder/prediction/{args.dataset_name}/modality_{args.modality}_lenc_{args.encoder_layers}_ldec_{args.fusion_encoder_layers}_alpha_{args.alpha}_beta_{args.beta}_train.csv'
       
    if args.eval:
        print('Start testing for one epoch !!!')
        checkpoint_path = args.eval
        checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load to CPU first
        model.load_state_dict(checkpoint["model"], strict=True)
        with torch.no_grad():
            test_stats, _, _ = evaluate( model, 0,  criterion, data_loader_val, device, logger, args, 0, data_type='test', results_path = test_results_path, nprocs=utils.get_world_size())  
            train_stats, _, _ = evaluate(model, 0,  criterion, data_loader_train, device, logger, args, 0, data_type='train' , results_path = train_results_path, nprocs=utils.get_world_size())   
    else:
        print("Start training")
        start_time = time.time()
        best_micro_f1_test, pre_micro_f1_test = 0, 0
        for epoch in range(args.start_epoch, args.epochs):
            best_micro_f1_test = max(best_micro_f1_test, pre_micro_f1_test)
            
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, logger, epoch,
                args.clip_max_norm, args.numclass, args.dataset_name)

            # extra checkpoint before LR drop and every 100 epochs
            lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                    
            test_stats, pre_micro_f1_test, best_epoch = evaluate(model, best_micro_f1_test,  criterion, data_loader_val, device, logger, args, epoch, data_type = "test", results_path = test_results_path, nprocs=utils.get_world_size())
            if(best_epoch):
                train_stats, _, _  = evaluate(model, best_epoch,  criterion, data_loader_train, device, logger, args, epoch, data_type = "train", results_path = train_results_path, nprocs=utils.get_world_size())

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log_tran&test.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Small model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)[args.dataset_name]
    args.train_video_id_set = data_info['train_video_id_set']
    args.test_video_id_set = data_info['test_video_id_set']
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
