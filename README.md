# SLICER: Sliced Fine-Tuning For Low-Resource Cross-Lingual Transfer for Named Entity Recognition

This is the implementation for training SLICER, a fine-tuning approach to enhance cross-lingual transfer to typologically distant languages for multilingual pre-trained languages models such as XLM-Roberta.

## Installation

You can install the required dependencies in two steps:

1. `conda env create -f environment.yaml`
2. Activate the conda environment `conda env activate trident_xtreme`
3. Change your working directory to `trident`
4. `pip install -e ./`

## Implementation

SLICER augments the training step of fine-tuning token-level classification tasks, which can be found in `training_step` of `RobustTokenClassification` in `trident-xtreme/src/projects/robust/module.py`.

```python
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # self.model: AutoModelForTokenClassification
        # self.model.roberta: RobertaEncoder
        # self.K = d/h as per paper
        embeds = self.model.roberta(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state
        labels = batch["labels"]
        mask = torch.logical_and(batch["attention_mask"] == 1, labels != -100)
        embeds = embeds[mask]
        embeds = self.model.dropout(embeds)
        w = self.model.classifier.weight.data.T
        # reshape classifier to d/h, h, |C|
        w = w.reshape(self.K, self.dim, -1)
        N, _ = embeds.shape
        # reshape embeds to N, d/h, |C|
        embeds = embeds.view(N, self.K, self.dim)
        # compute slice logits and reshape to (num slices, |C|)
        logits = (
            torch.einsum("ndk, dkl->ndl", embeds, w) + self.model.classifier.bias
        ).reshape(-1, self.model.num_labels)
        labels = labels[mask]
        # compute averaged loss over token-slices by computing loss by slice with correspondingly repeated labels
        loss = cross_entropy(logits, labels.repeat_interleave(self.K))
        self.log("train/loss", loss)
        return loss
```

As described in the paper, we slice both the token representations and classifier during training and propagate the averaged losses over token-slices.

## Reproduction

Then, you can execute our base experiments via 

1. CD into `trident_xtreme`
2. `bash wikiann.sh 42 10 0.00001` for standard fine-tuning or `bash wikiann_robust.sh 42 10 0.00001 1` for SLICER with `h=1`.

The positional arguments are as follows.

```bash
# seed, num epochs, learning rate, dim=h
bash wikiann_robust.sh 42 10 0.00001 1
```
# Contact

**Name:** Fabian David Schmidt\
**Mail:** fabian.schmidt@uni-wuerzburg.de\
**Affiliation:** Center For Artificial Intelligence and Data Science (CAIDAS), University of Würzburg


# TODO

- [ ] Link paper from ACL anthology
- [ ] Citation to be added once proceedings are released
