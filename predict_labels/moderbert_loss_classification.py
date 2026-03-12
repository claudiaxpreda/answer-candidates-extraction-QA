import sys
import pandas as pd
import ast
import time
import prepare_input_classification as pic
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.utils import shuffle


from transformers import (
    AutoModel,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    default_data_collator,
)

from datasets import Dataset
from dotenv import dotenv_values

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.metrics import f1_score

tokens = dotenv_values(".env")


MAX_LEN_SEQ = 512
MAX_LEN_BERT = 768
HUGGING_TOKEN = tokens["HUGGING_FACE_TOKEN_WRITE"]
ENCODER_MODEL_NAME = "answerdotai/ModernBERT-base"
API_KEY_WB = tokens["WANDB_KEY"]
LR = 2e-5
LABELS_NUM = 4
TRAIN_DATASET_PATH = "data/final/train_labeled_updated.csv"
VALIDATION_DATASET_PATH = "data/final/validation_labeled_updated.csv"
TEST_DATASET_PATH = "data/final/test_labeled_updated.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_slice(source_file):
    slice_df = pd.read_csv(
        source_file,
        keep_default_na=False,
        converters={"token_indexes": ast.literal_eval},
    )

    return slice_df


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=-1)
    labels = np.argmax(labels, axis=-1)
    mask = labels != 0
    labels = labels[labels != 0]
    predictions = predictions[mask]
    score = f1_score(labels, predictions, labels=labels, pos_label=1, average="micro")
    return {"f1": float(score) if score == 1 else score}


class CustomQAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            reference_compile=False,
            attn_implementation="eager",
        )
        self.config = self.encoder.config
        self.num_labels = LABELS_NUM
        self.dense1 = nn.Linear(768, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, LABELS_NUM)

    def forward(self, context_ids, attention_mask, context_masks, labels=None):
        outputs = self.encoder(input_ids=context_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state  # Shape: [B, 512, 768]

        # questions_ids shape: [B, MAX_LEN_PAD, 512]
        context_masks_cast = context_masks.to(embedding.dtype)
        product_step_a = torch.bmm(
            context_masks_cast, embedding
        )  # Shape: [B, MAX_LEN_PAD, 768]

        # Sum along the sequence dimension (dim=2) to find non-zero entries
        product_step_b = (context_masks_cast != 0).sum(dim=2, keepdim=True).float()

        product_step_c = product_step_a / torch.clamp(product_step_b, min=1.0)

        mask = (product_step_b > 0).squeeze(-1)

        layer1 = F.relu(self.dense1(product_step_c))

        layer1 = layer1 * mask.unsqueeze(-1).float()

        layer2 = self.dense2(layer1)

        logits = self.dense3(layer2)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # target_indices = labels

            target = torch.argmax(torch.tensor(labels), dim=2)
            target = target.view(-1)
            mask = target != 0
            target = target[mask]
            source = logits.view(-1, self.num_labels)[mask]

            loss = loss_fct(source, target)

        return {"loss": loss, "logits": logits} if loss is not None else logits


def main(max_len_pad, epochs, hub_name):
    print("Program start time:" + time.strftime("%H:%M:%S", time.localtime()))

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_NAME)
    mlb = MultiLabelBinarizer(classes=[1, 2, 3, 4])

    input_train_df = shuffle(read_slice(TRAIN_DATASET_PATH))
    input_validation_df = shuffle(read_slice(VALIDATION_DATASET_PATH))

    x_train, y_train, _ = pic.prepare_input_pt(
        input_train_df, tokenizer, int(max_len_pad), mlb, device
    )
    x_val, y_val, _ = pic.prepare_input_pt(
        input_validation_df, tokenizer, int(max_len_pad), mlb, device
    )

    train_dict = {
        "context_ids": x_train[0],
        "attention_mask": x_train[1],
        "context_masks": x_train[2],
        "labels": y_train,
    }

    val_dict = {
        "context_ids": x_val[0],
        "attention_mask": x_val[1],
        "context_masks": x_val[2],
        "labels": y_val,
    }

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    model = CustomQAModel(ENCODER_MODEL_NAME)
    wandb.login(key=API_KEY_WB)

    training_args = TrainingArguments(
        output_dir=hub_name,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=LR,
        num_train_epochs=int(epochs),
        bf16=False,  # bfloat16 training
        optim="adamw_torch_fused",  # improved optimizer
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        use_mps_device=False,
        metric_for_best_model="f1",
        # push to hub parameters
        push_to_hub=True,
        hub_strategy="every_save",
        hub_token=HUGGING_TOKEN,
        torch_compile=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )
    trainer.train()
    wandb.finish()

    print("Program end time:" + time.strftime("%H:%M:%S", time.localtime()))

    return 0

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
