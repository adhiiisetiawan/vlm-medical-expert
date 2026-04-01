import os
import re
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn import metrics

from m3_generator import M3Generator

from utils import (
    pad_and_reorder_sequence,
    get_labels
)

EXPERT_RE = re.compile(r'<([^<>]+)>')
MODEL_RE = re.compile(r'BRATS|CXR|VISTA3D')
PARAMETER_RE = re.compile(r'\((.*?)\)')

def get_expert_and_param(data:list[str]):
    expert = []
    param = []

    for d in data:
        p = PARAMETER_RE.search(d)
        e = MODEL_RE.search(d)

        expert.append(e.group(0) if e else "NONE")
        param.append(p.group(0) if p else "NONE")

    return expert, param

def main(args):
    generator = M3Generator(
        source=args.source,
        model_path=args.modelpath,
        conv_mode=args.convmode,
    )
    if args.moe:
        generator.insert_switch_layer()

    with open(args.data_dir) as f:
        samples = json.load(f)

    gt_result = []
    replay_result = []
    chat_samples = []

    # Get the chat samples for 1% of the sample (minimum 10)
    step = max(1, len(samples) // max(10, int(len(samples)*0.01)))
    
    SLICES_DIR = str(Path(args.data_dir).parent / "slices") + "/"

    # Var binding
    find_expert = EXPERT_RE.findall
    generate = generator.generate
    append_gt = gt_result.extend
    append_replay = replay_result.extend

    # Inference Loop
    for idx, sample in enumerate(tqdm(samples, desc="Processing sample", total=len(samples))):
        
        image_paths = [SLICES_DIR + img for img in sample["images"]]
        label_path = image_paths[-1]
        input_images = image_paths[:-1]

        # Generate response
        replay_msg, prompt_msg = generate(
            prompt=sample["conversations"][0]["value"],
            image_paths=input_images,
            max_new_tokens=1024
        )

        gt_msg = sample["conversations"][1]["value"]

        # Find expert model triggers
        expert_replay = find_expert(replay_msg)
        expert_gt = find_expert(gt_msg)

        expert_replay, expert_gt = pad_and_reorder_sequence(expert_replay, expert_gt)

        # Append to list
        append_replay(expert_replay)
        append_gt(expert_gt)

        # Log the messages
        if idx % step == 0:
            chat_samples.append({
                "question": prompt_msg,
                "gt": gt_msg,
                "replay": replay_msg
            })

    # Evaluation
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    gt_result, gt_params = get_expert_and_param(gt_result)
    replay_result, replay_params = get_expert_and_param(replay_result)

    labels = get_labels(gt_result, replay_result)

    report = metrics.classification_report(gt_result, replay_result, labels=labels, output_dict=True, zero_division=0)
    param_accuracy = metrics.accuracy_score(gt_params, replay_params)
    report["Parameter Accuracy"] = param_accuracy

    # Save result
    with open(os.path.join(args.output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    with open(os.path.join(args.output_dir, "chat_samples.json"), "w") as f:
        json.dump(chat_samples, f, indent=4)

    with open(os.path.join(args.output_dir, "expert_trigger_count.json"), "w") as f:
        json.dump(generator.get_expert_trigger_count(), f, indent=4)

    cm = metrics.confusion_matrix(gt_result, replay_result, labels=labels)

    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    with open(os.path.join(args.output_dir, "confusion_matrix.json"), "w") as f:
        json.dump({
            "classes": labels,
            "matrix": cm.tolist()
        }, f, indent=4)

    plt.figure(figsize=(10, 10))
    disp.plot()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))

if __name__ == "__main__":

    import torch

    print("Device name:", torch.cuda.get_device_name(0))
    print("CUDA_VISIBLE_DEVICES device count:", torch.cuda.device_count())
    if torch.cuda.device_count() != 1:
        print("! WARNING: CUDA_VISIBLE_DEVICES has not been properly defined !")

    load_dotenv()
    parser = argparse.ArgumentParser()
    # TODO: Add the argument to load multiple models from a JSON file
    parser.add_argument(
        "--convmode",
        type=str,
        default=os.getenv("VILA_CONV_MODE", "vicuna_v1"),
        help="The conversation mode to use. For 8B models, use 'llama_3'. For 3B and 13B models, use 'vicuna_v1'.",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        default=os.getenv("VILA_MODEL", "MONAI/Llama3-VILA-M3-3B"),
        help=(
            "The path to the model to load. "
            "If source is 'local', it can be '/data/checkpoints/vila-m3-8b'. If "
            "If source is 'huggingface', it can be 'MONAI/Llama3-VILA-M3-8B'."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("VILA_SOURCE", "huggingface"),
        help="The source of the model. Option is 'huggingface' or 'local'.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("DATA_DIR", "../data/experts/brats/llama_gen_expert_data_brats_test.json")
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("OUTPUT_DIR", "./output")
    )
    parser.add_argument(
        "--moe",
        type=bool,
        default=False
    )
    args = parser.parse_args()
    main(args)