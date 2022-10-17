import numpy as np
import torch
from pytorch_lightning import LightningModule


def entropy(s: torch.Tensor) -> torch.Tensor:
    sq = s.square().to("cpu")
    f = sq / sq.sum()
    denom = np.log(sq.shape[0])
    num = f * f.log()
    return (-num / denom).sum()


def process_outputs(module, outputs, batch, stage):
    embs = outputs["hidden_states"][-1]
    mask = torch.logical_and(batch["attention_mask"] == 1, batch["labels"] != -100)
    outputs["embs"] = embs[mask]
    return outputs


def store_data(module, outputs, stage):
    mask = torch.logical_and(outputs["attention_mask"] == 1, outputs["labels"] != -100)
    token_labels = outputs["labels"][mask]
    with open(f"{stage}.pt", "wb") as file:
        torch.save({"embs": outputs["embs"], "labels": token_labels}, file)
    return outputs


def compute_entropy(embs):
    _, s, _ = torch.linalg.svd(embs, full_matrices=False)
    s = s.to("cpu")
    e = entropy(s)
    return e


def average_token_self_attention(
    attentions: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    # b, heads, len, len
    batch_size, num_heads, seq_len, _ = attentions.shape
    diag_idx = torch.arange(seq_len, device=attentions.device)
    diag = attentions[:, :, diag_idx, diag_idx]
    avg_diag = (diag.mean(1) / attention_mask.sum(-1)[:, None]).sum(-1)
    return avg_diag


def demean_self_attentions(
    attentions: tuple[torch.Tensor], attention_mask: torch.Tensor
) -> torch.Tensor:
    return torch.mean(
        torch.stack(
            [average_token_self_attention(a, attention_mask) for a in attentions], 1
        ),
        -1,
    )


def eval_attention(
    module: LightningModule, outputs: dict, batch: dict, *args, **kwargs
):
    for layer in range(12):
        outputs[f"score_{layer}"] = average_token_self_attention(
            outputs["attentions"][layer], batch["attention_mask"]
        )

    return outputs
