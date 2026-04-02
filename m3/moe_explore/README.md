# Mixture of Experts Exploration
## Project Structure
```
routing_evaluation/
├── mingpt/                 <- MoE implementation on minGPT (The whole repo)
│   ├── mingpt/
│   │   ├── model.py            <- Modified minGPT code
│   │   └── ...
│   ├── demo.ipynb              <- Demo for the new architecture (from minGPT)
│   ├── generate.ipynb          <- Another demo for the new architecture (from minGPT)
│   └── ...
│
├── smolvlm/                <- MoE implementation on smolVLM
│   ├── moe_smolvlm.ipynb       <- MoE implementation on huggingface model
│   └── original_demo.ipynb     <- The original huggingface demo
│
├── vit_cifar100/           <- MoE implementation on ViT from scratch
│   ├── moevit.ipynb            <- MoE ViT training code
│   ├── switchvit.ipynb         <- Switch ViT training code
│   ├── vit.ipynb               <- ViT training code
│   └── transformer.py          <- File containing all of the architecture used
│
└── README.md
```

## Description
Note: If you want to look for the latest switch layer code, please look at `m3/routing_evaluation/utils.py`

### vit_cifar100
Implementation of MoE and Switch Transformer on ViT that is made from scratch. Comparing MoE ViT, Switch ViT and regular ViT when used for training with CIFAR100 dataset.

A `get_expert_trigger_count` and `reset_expert_trigger_count` is also available on both `MoEViT` and `SwitchViT` class to obtain all the expert model triggering count and to reset the expert model triggering count to look for model collapse.

### minGPT
Implementation of MoE on minGPT, a gpt2 architecture. Modification is made on `mingpt/model.py` file inside the `__init__` and `from_pretrained` method from the `GPT` class.

A `get_expert_trigger_count` and `reset_expert_trigger_count` is also available on `GPT` class to obtain all the expert model triggering count and to reset the expert model triggering count to look for model collapse.

### smolVLM
Implementation of MoE on smolVLM, a model loaded directly from huggingface. Implementation is done by directly modifying the objects inside the model class. 

Do note that the huggingface version for VILA M3 environtment might not be compatible with the smolVLM code.

### VILA M3
Implementation of MoE on VILA M3. Do note that the code exists on a different folder (`m3/routing_evaluation/`). Implementation is done similarly as the smolVLM one, but it is now a method inside the `m3_generator` class and can be enabled optionally via the `insert_switch_layer` method. 

A `get_expert_trigger_count` method is also available to obtain all the expert model triggering count to look for model collapse. 

`reset_expert_trigger_count` has not yet been implemented and a safeguard if `get_expert_trigger_count` and `reset_expert_trigger_count` are used without any switch layer insertion does not exists so please use them with caution.