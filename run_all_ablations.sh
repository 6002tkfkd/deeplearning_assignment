#!/bin/bash
# 중단 후 재실행 시 완료된 실험은 자동으로 건너뜁니다.

mkdir -p ./results
echo "[$(date)] ===== Validating all configs first... ====="
python validate_configs.py
if [ $? -ne 0 ]; then
  echo "[ERROR] Config validation failed. Fix errors above before running."
  exit 1
fi
echo "[$(date)] ===== All configs valid. Starting ablations ====="

declare -A CONFIGS=(
  ["ablation_augmentation"]="ablation_augmentation"
  ["ablation_projection"]="ablation_projection"
  ["ablation_batchsize"]="ablation_batchsize"
  ["ablation_temperature"]="ablation_temperature"
  ["ablation_backbone"]="ablation_backbone"
  ["ablation_loss"]="ablation_loss"
)

for group in ablation_augmentation ablation_projection ablation_batchsize \
             ablation_temperature ablation_backbone ablation_loss; do

  SAVE_DIR="./results/${group}"
  mkdir -p "$SAVE_DIR"

  echo ""
  echo "========================================"
  echo "[$(date)] Group: ${group}"
  echo "  save_dir: ${SAVE_DIR}"
  echo "========================================"

  python run_experiment.py \
    --mode ablation \
    --config "experiments/configs/${group}.yaml" \
    --base_config experiments/configs/base.yaml \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee -a "$SAVE_DIR/${group}.log"

  echo "[$(date)] Done: ${group}"
done

echo ""
echo "[$(date)] ===== All ablations complete! ====="

# 전체 결과 요약 차트 생성
echo "[$(date)] Generating summary report..."
python run_experiment.py --mode report --results_dir ./results
echo "[$(date)] Done."
