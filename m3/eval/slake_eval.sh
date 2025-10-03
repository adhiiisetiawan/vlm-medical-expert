cd /mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/VILA
export PYTHONPATH=/mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/VILA
echo "PYTHONPATH is $PYTHONPATH"


DATA_PATH="/mnt/extended-home/adhi/dataset/slake/Slake1.0/test_instruct.json"
IMAGE_DIR="/mnt/extended-home/adhi/dataset/slake/Slake1.0/imgs"
OUTPUT_ROOT="/mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/results/slakevqa"
OUTPUT_PATH="$OUTPUT_ROOT/outputs.jsonl"
RESULT_PATH="$OUTPUT_ROOT/results.json"
CONV_MODE="vicuna_v1"
MODEL_PATH="/mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/checkpoint/Llama3-VILA-M3-3B"


python -m llava.eval.model_vqa_science \
        --model-path $MODEL_PATH \
        --question-file $DATA_PATH \
        --image-folder $IMAGE_DIR \
        --answers-file $OUTPUT_PATH \
        --num-chunks 1 \
        --chunk-idx 0 \
        --single-pred-prompt \
        --conv-mode $CONV_MODE \
        # --generation-config "$GENERATION_CONFIG" \


python /mnt/extended-home/adhi/projects/VLM-Radiology-Agent-Framework/m3/eval/scripts/metric_slakevqa.py \
	--input $DATA_PATH \
	--answers $OUTPUT_PATH \
	--output $RESULT_PATH \