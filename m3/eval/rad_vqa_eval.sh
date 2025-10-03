cd /mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/VILA
export PYTHONPATH=/mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/VILA

echo "PYTHONPATH is $PYTHONPATH"

DATA_PATH="/mnt/extended-home/adhi/dataset/vqa_rad/radvqa_test_instruct.json"
IMAGE_DIR="/mnt/extended-home/adhi/dataset/vqa_rad/images"
OUTPUT_ROOT="/mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/results/radvqa"
OUTPUT_PATH="$OUTPUT_ROOT/outputs.jsonl"
RESULT_PATH="$OUTPUT_ROOT/results.json"
GENERATION_CONFIG='{"max_new_tokens":1024,"do_sample":false,"temperature":0}'
CONV_MODE="vicuna_v1"
MODEL_PATH="/mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/checkpoint/Llama3-VILA-M3-3B"

export CUDA_VISIBLE_DEVICES=1


# python3 -m llava.eval.model_vqa_science \
#         --model-path $MODEL_PATH \
#         --question-file $DATA_PATH \
#         --image-folder $IMAGE_DIR \
#         --answers-file $OUTPUT_PATH \
#         --num-chunks 1 \
#         --chunk-idx 0 \
#         --single-pred-prompt \
#         --conv-mode $CONV_MODE
#         # --generation-config "$GENERATION_CONFIG" \ 


python3 /mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/m3/eval/scripts/metric_radvqa.py \
	--input $DATA_PATH \
	--answers $OUTPUT_PATH \
	--output $RESULT_PATH \