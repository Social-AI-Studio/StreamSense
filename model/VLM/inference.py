import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
from sklearn.metrics import f1_score, accuracy_score


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM Inference for Multimodal Sentiment Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data_type", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--dataset_name", type=str, default="MOSI", choices=["MOSI", "MOSEI", "HateBit"])
    parser.add_argument("--modality", type=str, default="vt", choices=["t", "v", "vt"])
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=25)
    parser.add_argument("--variance_threshold", type=int, default=100, help="Image variance threshold")
    return parser.parse_args()


class VLMInferenceConfig:
    def __init__(self, args: argparse.Namespace):
        self.data_type = args.data_type
        self.dataset_name = args.dataset_name
        self.modality = args.modality
        self.base_model_id = args.base_model_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.max_new_tokens = args.max_new_tokens
        self.root = args.root
        self.variance_threshold = args.variance_threshold
        self.dataset_path = f"{self.root}/dataset/{self.dataset_name}/VLM_input/{self.data_type}.csv"
        self.checkpoint_dir = f"{self.root}/model/VLM/checkpoint/{self.dataset_name}"
        self.output_dir = f"{self.root}/model/VLM/prediction/{self.dataset_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        if self.dataset_name == "HateBit":
            self.pos_token_id = 4699
            self.neg_token_id = 12484
            self.pos_label = "Offensive"
            self.neg_label = "Normal"
        else:
            self.pos_token_id = 36590
            self.neg_token_id = 39589
            self.pos_label = "Positive"
            self.neg_label = "Negative"

        self.black_image_path = "/mnt/sda/wanghan/ICLR2026/final_model/dataset/demo.png"


class VLMInference:
    def __init__(self, config: VLMInferenceConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None

    def _load_model(self, checkpoint_path: str = None):
        print(f"Loading base model: {self.config.base_model_id}")
        base_model = MllamaForConditionalGeneration.from_pretrained(
            self.config.base_model_id,
            torch_dtype=torch.float16
        ).to(self.config.device)
        base_model.generation_config.temperature = None
        base_model.generation_config.top_p = None
        self.processor = AutoProcessor.from_pretrained(self.config.base_model_id)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            print("Using base model without checkpoint.")
            self.model = base_model
            self.tokenizer = None

    def check_image_content(self, image_path: str) -> bool:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Image could not be loaded.")
            return np.var(image) > self.config.variance_threshold
        except Exception as e:
            print(f"Error checking image: {e}")
            return False

    def _create_prompt(self, text: str = "") -> str:
        if self.config.modality.upper() == "T":
            prompt_text = f"Text: '{text}'"
        elif self.config.modality.upper() == "V":
            prompt_text = "Text: ''"
        else:
            prompt_text = f"Text: '{text}'"
        return (f"{prompt_text} \nAnalyze the image and text to determine if the content is "
                f"{self.config.pos_label}. Respond with '{self.config.pos_label}' or '{self.config.neg_label}'.")

    def _prepare_messages_and_images(self, row: Dict, prompt: str):
        images = []
        if "V" in self.config.modality.upper():
            image_path = row.get('image', self.config.black_image_path)
        else:
            image_path = self.config.black_image_path
        image = Image.open(image_path).convert('RGB').resize((560, 560))
        images.append(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ]
        }]
        return messages, images

    @torch.no_grad()
    def inference_single(self, row: Dict):
        text = row.get('text', '')
        prompt = self._create_prompt(text)
        messages, images = self._prepare_messages_and_images(row, prompt)
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images, text=input_text, return_tensors="pt").to(self.config.device)
        prompt_length = inputs.input_ids.shape[-1]

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True
        )

        logits = output.logits
        output_sequences = output.sequences

        if self.tokenizer:
            response = self.tokenizer.decode(output_sequences[0][prompt_length:], skip_special_tokens=True)
        else:
            response = self.processor.decode(output_sequences[0][prompt_length:], skip_special_tokens=True)

        pos_logit = logits[0][0][self.config.pos_token_id]
        neg_logit = logits[0][0][self.config.neg_token_id]
        prob_vector = F.softmax(torch.stack([pos_logit, neg_logit], dim=0), dim=0)

        return {
            'video_id': row.get('video_id'),
            'cur_time_id': row.get('cur_time_id'),
            'prob_pos': prob_vector[0].item(),
            'prob_neg': prob_vector[1].item(),
            'pred': response,
            'gt': row.get('label')
        }

    def run_epoch_inference(self, checkpoint_path: str):
        self._load_model(checkpoint_path)
        try:
            dataset = load_dataset('csv', data_files=self.config.dataset_path)
            data_list = dataset['train']
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return [], 0.0, 0.0

        all_results = []
        y_true, y_pred = [], []

        for row in tqdm(data_list, desc=f"Inference {os.path.basename(checkpoint_path)}"):
            if row.get('label') is None:
                continue
            result = self.inference_single(row)
            all_results.append(result)
            pred_label = 0 if result['pred'] == self.config.neg_label else 1
            y_pred.append(pred_label)
            y_true.append(0 if row['label'] == self.config.neg_label else 1)

        acc = accuracy_score(y_true, y_pred)
        m_f1 = f1_score(y_true, y_pred, average='macro')
        return all_results, acc, m_f1

    def run_all_checkpoints(self):
        checkpoint_dirs = sorted(
            [str(p) for p in Path(self.config.checkpoint_dir).glob("checkpoint-*") if p.name.split('-')[-1].isdigit()],
            key=lambda x: int(x.split('-')[-1])
        )

        best_mf1 = -1.0
        best_results = []
        best_checkpoint = ""

        for ckpt in checkpoint_dirs:
            print(f"\n--- Running inference for checkpoint: {ckpt} ---")
            results, acc, mf1 = self.run_epoch_inference(ckpt)
            print(f"Checkpoint {ckpt} | Acc: {acc:.4f} | M-F1: {mf1:.4f}")

            if mf1 > best_mf1:
                best_mf1 = mf1
                best_results = results
                best_checkpoint = ckpt
                # Save best results
                output_file = os.path.join(self.config.output_dir, f"modality_{self.config.modality}_{self.config.data_type}.csv")
                pd.DataFrame(best_results).to_csv(output_file, index=False)
                # Save best checkpoint path
                best_ckpt_path = os.path.join(self.config.checkpoint_dir, "checkpoint-best")
                if os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                os.symlink(best_checkpoint, best_ckpt_path)

        print(f"\nBest checkpoint: {best_checkpoint} | M-F1: {best_mf1:.4f}")
        print(f"Best inference results saved to: {output_file}")


def main():
    args = parse_arguments()
    config = VLMInferenceConfig(args)

    if not os.path.exists(config.dataset_path):
        print(f"Dataset not found: {config.dataset_path}")
        sys.exit(1)

    engine = VLMInference(config)
    engine.run_all_checkpoints()


if __name__ == "__main__":
    main()
