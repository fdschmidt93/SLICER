import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModel
from trident import TridentModule


@dataclass
class TokenClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    labels: Optional[torch.Tensor] = None


class Collator:
    def __init__(self, pretrained_model_name_or_path: str) -> None:
        from transformers import AutoTokenizer
        from transformers.data.data_collator import \
            DataCollatorForTokenClassification

        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        collate_fn = DataCollatorForTokenClassification(
            tokenizer=tokenizer, padding=True, max_length=510
        )
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn

    def __call__(self, inputs):
        batch = self.collate_fn(inputs)
        batch["output_hidden_states"] = True
        return batch


class RobustTokenClassification(TridentModule):
    def __init__(
        self,
        dim: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim if isinstance(dim, int) else 768
        self.K = 768 // self.dim

    def setup(self, stage: str):
        super().setup(stage)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        embeds = self.model.roberta(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state
        labels = batch["labels"]
        mask = torch.logical_and(batch["attention_mask"] == 1, labels != -100)
        embeds = embeds[mask]
        embeds = self.model.dropout(embeds)
        w = self.model.classifier.weight.data.T
        w = w.reshape(self.K, self.dim, -1)
        N, _ = embeds.shape
        embeds = embeds.view(N, self.K, self.dim)
        logits = (
            torch.einsum("ndk, dkl->ndl", embeds, w) + self.model.classifier.bias
        ).reshape(-1, self.model.num_labels)
        labels = labels[mask]
        loss = cross_entropy(logits, labels.repeat_interleave(self.K))
        self.log("train/loss", loss)
        return loss
