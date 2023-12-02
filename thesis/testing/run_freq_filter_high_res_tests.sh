PROJECT_NAME="thesis-freq-filter-high-res-tests"
CONFIG_PATH="/Users/aidan/projects/engsci/year4/thesis/implementation/thesis/testing/configs/base_config.yaml"

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "h=5" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 5 \
    # --low-freq-cutoff-override 1 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=5 h=10" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 10 \
    --low-freq-cutoff-override 5 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=10 h=15" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 15 \
    --low-freq-cutoff-override 10 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=15 h=20" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 20 \
    --low-freq-cutoff-override 15 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=20 h=25" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 25 \
    --low-freq-cutoff-override 20 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=25 h=30" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 30 \
    --low-freq-cutoff-override 25 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=30 h=35" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 35 \
    --low-freq-cutoff-override 30 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=35 h=40" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 40 \
    --low-freq-cutoff-override 35 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=40 h=45" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 45 \
    --low-freq-cutoff-override 40 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "None" \
    --sample-rate-override 120 \