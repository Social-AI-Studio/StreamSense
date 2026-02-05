import argparse
import numpy as np

def str2bool(string):
	return True if string.lower() == 'true' else False

def get_args_parser():
	parser = argparse.ArgumentParser('Set IDU Online Detector', add_help=False)
	# fixed
	parser.add_argument('--lr', default=1e-4, type=float)     # 1e-4
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--weight_decay', default=1e-4, type=float)
	parser.add_argument('--epochs', default=10, type=int)
	parser.add_argument('--resize_featuree', default=False, type=str2bool, help='run resize prepare_data or not')
	parser.add_argument('--lr_drop', default=1, type=int)
	parser.add_argument('--clip_max_norm', default=1., type=float,
						help='gradient clipping max norm')  # dataparallel
	parser.add_argument('--dataparallel', action='store_true', help='multi-gpus for training')
	parser.add_argument('--removelog', action='store_true', help='remove old log')
	parser.add_argument('--version', default='v3', type=str,
						help="fixed or learned")  
	parser.add_argument('--world_size', default=1, type=int,
						help='number of distributed processes')
	parser.add_argument('--dist_url', default='tcp://127.0.0.1:12342', help='url used to set up distributed training')
	parser.add_argument('--num_workers', default=8, type=int)
	parser.add_argument('--classification_h_loss_coef', default=1, type=float)
	parser.add_argument('--decoder_embedding_dim', default=768, type=int,   # 768
						help="decoder_embedding_dim")
	parser.add_argument('--decoder_embedding_dim_out', default=768, type=int,  # 256 512 768
						help="decoder_embedding_dim_out")
	parser.add_argument('--decoder_attn_dropout_rate', default=0.1, type=float,  # 0.1=0.2
						help="rate of decoder_attn_dropout_rate")
	parser.add_argument('--decoder_num_heads', default=8, type=int,  # 8 4
						help="decoder_num_heads")
	parser.add_argument('--lr_backbone', default=1e-4, type=float,    # 2e-4
						help="lr_backbone")
	parser.add_argument('--feature', default=None, type=str,
						help="feature type")
	parser.add_argument('--dim_feature', default=3072, type=int,
						help="input feature dims")
	parser.add_argument('--patch_dim', default=1, type=int,
						help="input feature dims")
	parser.add_argument('--embedding_dim', default=768, type=int,  # 768
						help="input feature dims")
	parser.add_argument('--num_heads', default=8, type=int,
						help="input feature dims")
	parser.add_argument('--attn_dropout_rate', default=0.1, type=float,
						help="attn dropout")
	parser.add_argument('--positional_encoding_type', default='learned', type=str,
						help="fixed or learned")  # learned  fixed
	parser.add_argument('--hidden_dim', default=768, type=int,  # 512 768
						help="Size of the embeddings")
	parser.add_argument('--dropout_rate', default=0.1, type=float,
						help="Dropout applied ")
	parser.add_argument('--margin', default=1., type=float)
	parser.add_argument('--dataset_file', type=str, default='../../dataset/data_split.json')
	parser.add_argument('--frozen_weights', type=str, default=None)
	parser.add_argument('--remove_difficult', action='store_true')
	parser.add_argument('--resume', default='', help='resume from checkpoint')
	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
						help='start epoch')
	parser.add_argument('--device', default='cuda:0',
						help='device to use for training / testing')

	# Transformer Structure
	parser.add_argument('--encoder_layers', default=2, type=int,
						help="num of encoder layers")
	parser.add_argument('--fusion_encoder_layers', default=2, type=int,
						help="Number of fusion_encoder_layers")
	
	# Transformer Loss
	parser.add_argument('--alpha', default=0.25, type=float)  # 1
	parser.add_argument('--beta', default=1.0, type=float) 

	# Dataset
	parser.add_argument('--numclass', default=2, type=int,
						help="Number of class")
	parser.add_argument('--dataset_name', default="MOSI", type=str,
						help="Name of dataset")
	parser.add_argument('--modality', default="vta", type=str, 
						help='the modality model support')
	# General
	parser.add_argument('--seed', default=10, type=int) # 10, 20, 30
	parser.add_argument('--s', default=1, type=int, metavar='N',
						help='the feature sample interval in seconds')
	parser.add_argument('--N', default=32, type=int,
						help="context window size in seconds")
	parser.add_argument('--output_dir', default='models',
						help='path where to save, empty for no saving')
	parser.add_argument('--eval', default="", type=str, help='checkpoint to evaluate')
	parser.add_argument('--root', default="", type=str, help='root path')
	
	
	return parser

