#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# NOTES
# dim refers to h in the paper
# max_epochs is always set to 10 in our experiments
# lr is one of 0.000005 0.00001 0.00002
# seed is one of 42 43 44 45 46 47 48 49 50 51
python run.py experiment=wikiann_zs seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32 test_after_training=true logger.wandb.project="wikiann-zs-repro" 'logger.wandb.name="seed=${seed}-lr=${module.optimizer.lr}-epochs=${trainer.max_epochs}-h=standard_ft"' source_lang=en
