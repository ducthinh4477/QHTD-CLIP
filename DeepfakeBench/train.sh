#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DETECTOR_PATH="./training/config/detector/effort_asy.yaml"

# Update these arrays to match the datasets prepared in dataset_json_folder.
TRAIN_DATASETS=(
	"FaceForensics++"
)

TEST_DATASETS=(
	"Celeb-DF-v2"
	"FaceShifter"
	"DeeperForensics-1.0"
)

python training/train.py \
	--detector_path "${DETECTOR_PATH}" \
	--train_dataset "${TRAIN_DATASETS[@]}" \
	--test_dataset "${TEST_DATASETS[@]}"
