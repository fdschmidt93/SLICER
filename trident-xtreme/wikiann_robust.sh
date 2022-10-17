#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# seed max epochs lr dim
# NOTES
# dim refers to h in the paper
# max_epochs is always set to 10 in our experiments
# lr is one of 0.000005 0.00001 0.00002
# seed is one of 42 43 44 45 46 47 48 49 50 51
python run.py experiment=wikiann_zs_robust seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.dim=${4} module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32 source_lang=en
