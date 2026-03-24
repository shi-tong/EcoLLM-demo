---
library_name: peft
license: other
base_model: /root/models/Qwen3-8B
tags:
- base_model:adapter:/root/models/Qwen3-8B
- llama-factory
- lora
- transformers
- unsloth
pipeline_tag: text-generation
model-index:
- name: lca_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lca_lora

This model is a fine-tuned version of [/root/models/Qwen3-8B](https://huggingface.co//root/models/Qwen3-8B) on the lca_data dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7530

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_8BIT with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.9.1+cu128
- Datasets 2.16.1
- Tokenizers 0.22.1