
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer import TransformerModel
from ipdb import set_trace
import timm
from .PositionalEncoding import (
    LearnedPositionalEncoding,
)
__all__ = ['visual_B16', 'visual_B32', 'visual_L16', 'visual_L32', 'visual_H14']
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import loralib as lora
class VisionTransformer_v3(nn.Module):
    def __init__(
        self,
        args,
        inp_dim,
        step,
        patch_dim,
        out_dim,
        embedding_dim,
        num_heads,
        encoder_layers,
        fusion_encoder_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=False,
        classification_pred_loss_coef=0.0,
        positional_encoding_type="learned", 
        num_channels=3072,
    ):
        super(VisionTransformer_v3, self).__init__()

        # Validation checks
        assert embedding_dim % num_heads == 0
        assert inp_dim % patch_dim == 0
        
        # Store configuration parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation
        self.classification_pred_loss_coef = classification_pred_loss_coef
        self.out_dim = out_dim
        
        # Calculate patches and dimensions
        inp_dim_after_sample = sorted(list(range(inp_dim - 1, -1, -step)))
        self.num_patches = int(len(inp_dim_after_sample) // patch_dim)
        self.fusion_encoder_layers = fusion_encoder_layers
        self.encoder_layers = encoder_layers

        # Initialize module lists for different components
        self.linear_encoding = nn.ModuleList()
        self.cls_token = nn.ParameterList()
        self.position_encoding = nn.ModuleList()
        self.pe_dropout = nn.ModuleList()
        self.encoder = nn.ModuleList()
        self.pre_head_ln = nn.ModuleList()

        # Map modality codes to full names
        modality_map = {"v": "visual", "t": "text", "a": "audio"}
        self.features = [modality_map[char] for char in args.modality]
        
        # Add fusion feature if multiple modalities
        if len(self.features) > 1:
            fusion_modality = "".join([char + '_' for char in self.features])
            self.features.append(f"{fusion_modality}fus")

        # Define feature dimensions for different modalities
        feature_dims = {
            "visual": 1024,
            "text": 1024,
            "audio": 1024,
            "visual_text_fus": 768,
            "visual_audio_fus": 768,
            "text_audio_fus": 768,
            "visual_text_audio_fus": 768,
        }

        # Build modules for each feature
        for feature in self.features:
            if feature in feature_dims:
                flatten_dim = feature_dims[feature]
            
            self.linear_encoding.append(nn.Linear(flatten_dim, embedding_dim))

            if "fus" in feature:
                self.seq_length = (self.num_patches) * len(args.modality) + 1
            else:
                self.seq_length = self.num_patches

            if ("fus" in feature and self.fusion_encoder_layers > 0) or self.encoder_layers > 0:
                self.position_encoding.append(
                    LearnedPositionalEncoding(
                        self.seq_length, 
                        self.embedding_dim, 
                        self.seq_length
                    )
                )
                self.pe_dropout.append(nn.Dropout(p=self.dropout_rate))
                self.pre_head_ln.append(nn.LayerNorm(embedding_dim))
                self.cls_token.append(nn.Parameter(torch.zeros(1, 1, embedding_dim)))
                self.encoder.append(
                    TransformerModel(
                        embedding_dim,
                        fusion_encoder_layers,
                        num_heads,
                        hidden_dim,
                        self.dropout_rate,
                        self.attn_dropout_rate,
                    )
                )
            else:
                # Add None placeholders when no encoder is used
                self.position_encoding.append(None)
                self.pe_dropout.append(None)
                self.pre_head_ln.append(None)
                self.cls_token.append(None)
                self.encoder.append(None)

        # Final projection and classification layers
        self.projections = nn.Linear(embedding_dim, embedding_dim)
        self.mlp_head = nn.Linear(embedding_dim, out_dim)

    def forward(self, sequence_input_visual, sequence_input_text, sequence_input_audio):
        x_dict = {}
        # Unimodal Encoder
        for feature in self.features:
            if feature == "visual":
                x_dict[feature] = sequence_input_visual
            elif feature == "text":
                x_dict[feature] = sequence_input_text
            elif feature == "audio":
                x_dict[feature] = sequence_input_audio
            else:
                continue
            
            feature_idx = self.features.index(feature)
            x_dict[feature] = self.linear_encoding[feature_idx](x_dict[feature])
            
            if self.encoder_layers > 0:
                x_dict[feature] = self.position_encoding[feature_idx](x_dict[feature])
                x_dict[feature] = self.pe_dropout[feature_idx](x_dict[feature])
                x_dict[feature] = self.encoder[feature_idx](x_dict[feature])
                x_dict[feature] = self.pre_head_ln[feature_idx](x_dict[feature])
        
        # Create feature embeddings by averaging across sequence dimension
        feature_emb = {modality: x_dict[modality].mean(dim=1) for modality in x_dict}
        for modality in ["text", "audio", "visual"]:
            feature_emb.setdefault(modality, None)

        # Multimodal Fusion Encoder
        output_x = []
        for feature in self.features:
            if len(self.features) == 1: # No fusion for unimodal
                output_x = [x_dict[feature].mean(dim=1)]

            elif "fus" in feature:
                component_modalities = [x for x in feature.split("_")   if x and x not in ["fus"]]
                x = torch.cat([x_dict[comp_feature] for comp_feature in component_modalities], dim=1)
                if self.fusion_encoder_layers > 0:
                    feature_idx = self.features.index(feature)
                    x = self.linear_encoding[feature_idx](x)
                    cls_tokens = self.cls_token[feature_idx].expand(x.shape[0], -1, -1)
                    x = torch.cat((x, cls_tokens), dim=-2)
                    
                    x = self.position_encoding[feature_idx](x)
                    x = self.pe_dropout[feature_idx](x)
                    x = self.encoder[feature_idx](x)
                    x = self.pre_head_ln[feature_idx](x)
                    output_x.append(x[:, -1])

     
        # Concatenate all outputs and apply MLP layers
        output_x = torch.cat(output_x, dim=1)
        labels = self.mlp_head(self.projections(output_x))
        
        return labels, feature_emb