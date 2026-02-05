import argparse
import os
import sys
from typing import Dict, List
import numpy as np
import torch
from torch import nn
from PIL import Image
import cv2
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for VLM fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaMA-VLM for multimodal classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_name", type=str, default="MOSI",choices=["MOSI", "MOSEI", "HateBit"], help="Dataset name (e.g., MOSI, MOSEI, HateBit)")
    parser.add_argument("--modality", type=str, default="VT", choices=["V", "T", "VT"], help="Modality type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--refinetune", type=lambda x: x.lower() == 'true', default=False, help="Set to True to resume fine-tuning from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training")
    parser.add_argument("--root", type=str, default=".", help="Root path for data and checkpoints")

    return parser.parse_args()


class VLMTrainerConfig:
    """Configuration class for VLM fine-tuning."""
    def __init__(self, args: argparse.Namespace):
        self.dataset_name = args.dataset_name
        self.modality = args.modality
        self.epochs = args.epochs
        self.refinetune = args.refinetune
        self.device = args.device
        self.root = args.root

        self.dataset_path =  f"{self.root}/dataset/{self.dataset_name}/VLM_input/train.csv"
        self.output_dir = f"{self.root}/model/VLM/checkpoint/{self.dataset_name}/"

        # Model setup
        self.base_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.model_id = self.output_dir if self.refinetune else self.base_model_id


class VLMTrainer:
    """Trainer class for LLaMA-VLM fine-tuning."""
    def __init__(self, config: VLMTrainerConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.prompt = self._build_prompt()
        self._prepare_model()

    def _build_prompt(self) -> str:
        """Generate prompt templates based on dataset and modality."""
        if self.config.dataset_name == "HateBit":
            pos_word, neg_word = "Offensive", "Normal"
        else:
            pos_word, neg_word = "Positive", "Negative"

        if self.config.modality == "V":
            return f"Analyze the image to determine if the content is {pos_word}. Respond with '{pos_word}' or '{neg_word}'—just one word, no sentences."
        elif self.config.modality == "T":
            return f"Text: '{{text}}' \n Analyze the text to determine if the content is {pos_word}. Respond with '{pos_word}' or '{neg_word}'—just one word, no sentences."
        else:
            return f"Text: '{{text}}' \n Analyze the image and text to determine if the content is {pos_word}. Respond with '{pos_word}' or '{neg_word}'—just one word, no sentences."

    @staticmethod
    def check_image_content(image_path: str, variance_threshold: int = 100) -> bool:
        """Check if image has enough variance to be considered valid."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image could not be loaded.")
        return np.var(image) > variance_threshold

    def format_data(self, sample: Dict) -> Dict:
        """Format data sample into message-style input."""
        modality = self.config.modality
        if "V" in modality:
            user_content = {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": self.prompt.format(text=sample["text"])},
                ],
            }
        else:
            user_content = {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{self.config.root}/data/black_img.png"},
                    {"type": "text", "text": self.prompt.format(text=sample["text"])},
                ],
            }

        assistant_content = {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        }

        return {"input_ids": [user_content, assistant_content, 1.0]}

    def collate_fn(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collator for TRL SFTTrainer."""
        batch = {}
        for sample in examples:
            messages = sample["input_ids"][:2]
            labels = ""
            image = None

            for message in messages:
                role = message["role"]
                for content in message["content"]:
                    if content["type"] == "text" and role == "assistant":
                        labels += content["text"] + " "
                    if content["type"] == "image" and role == "user":
                        image_path = content["image"]
                        image = Image.open(image_path).convert("RGB").resize((560, 560))

            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(image, input_text, size={"height": 560, "width": 560},
                                    padding="max_length", truncation=True, max_length=512,
                                    return_tensors="pt").to(self.config.device)

            labels = inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
            labels[labels == image_token_id] = -100
            inputs["labels"] = labels
            inputs["iou"] = sample["input_ids"][2]


            for key, val in inputs.items():
                if key not in batch:
                    batch[key] = [val.squeeze(0) if key != "iou" else torch.tensor(inputs[key])]
                else:
                    batch[key].append(val.squeeze(0) if key != "iou" else torch.tensor(inputs[key]))

        for key in batch:
            batch[key] = torch.stack(batch[key])
        return batch

    def _prepare_model(self):
        """Load processor, model, and apply LoRA."""
        print(f"Loading model from {self.config.model_id}")

        self.processor = AutoProcessor.from_pretrained(self.config.base_model_id, max_length=512, return_tensors="pt")
        self.tokenizer = self.processor.tokenizer

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.config.model_id,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=self.config.device,
            quantization_config=bnb_config,
        )

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        self.peft_config = peft_config

    def run_training(self):
        """Execute the fine-tuning pipeline."""
        dataset = load_dataset('csv', data_files=self.config.dataset_path, split="train").shuffle(seed=42)
        train_dataset = [self.format_data(s) for s in dataset if s["label"] and s["type"] != "test"]
        valid_dataset = [self.format_data(s) for s in dataset if s["label"] and s["type"] != "train"]

        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}", flush=True)

        args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=5,
            save_strategy="epoch",
            learning_rate=2e-4,
            max_grad_norm=0.3,
            dataloader_pin_memory=False,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            push_to_hub=True,
            report_to="tensorboard",
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self.collate_fn,
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
        )

        trainer.train(resume_from_checkpoint=self.config.refinetune)
        trainer.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"Training complete. Model saved to {self.config.output_dir}")


def main():
    """Main entry point for training."""
    args = parse_arguments()
    config = VLMTrainerConfig(args)

    try:
        trainer = VLMTrainer(config)
        trainer.run_training()
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
