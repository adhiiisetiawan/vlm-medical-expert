# Evaluation

## Project Structure
```
routing_evaluation/
├── dummy_dataset/      <- Dummy dataset for local testing
│   ├── mri_Brats18_2013_31_1/
│   └── datalist.json
├── .env
├── evaluation.py       <- Evaluation code
├── m3_generator.py     <- Model class
├── README.md
└── utils.py            <- Utility functions
```

## Environtment setup
Setup an .env file with this specifications. Parameter values can be modified according to requirements.
```env
VILA_MODEL=MONAI/Llama3-VILA-M3-3B
VILA_SOURCE=huggingface
VILA_CONV_MODE=vicuna_v1

DATA_DIR=../data/experts/brats/llama_gen_expert_data_brats_test.json
OUTPUT_DIR=./output
```

## Running the evaluation
For BRATS data, a dummy dataset is provided for a local run. Run the dummy data trough brats expert model data preparation before using it.


To run the evaluation script, use the following command:
```
python3 evaluation.py
```

## MoE Integration
MoE integration can be done trough `--moe True` parameter when running `evaluation.py`

There is now a method inside the `m3_generator` class and can be enabled optionally via the `insert_switch_layer` method. 

A `get_expert_trigger_count` method is also available to obtain all the expert model triggering count to look for model collapse. 

`reset_expert_trigger_count` has not yet been implemented and a safeguard if `get_expert_trigger_count` and `reset_expert_trigger_count` are used without any switch layer insertion does not exists so please use them with caution.