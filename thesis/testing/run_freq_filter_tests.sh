PROJECT_NAME="thesis-freq-filter-tests"
CONFIG_PATH="/Users/aidan/projects/engsci/year4/thesis/implementation/thesis/testing/configs/base_config.yaml"

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "h=10" \
    --sample-rate-override 120 \
    --high-freq-cutoff-override 10 \
    # --low-freq-cutoff-override 1 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=5 h=15" \
    --sample-rate-override 120 \
    --low-freq-cutoff-override 5 \
    --high-freq-cutoff-override 15 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=10 h=20" \
    --sample-rate-override 120 \
    --low-freq-cutoff-override 10 \
    --high-freq-cutoff-override 20 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=15 h=25" \
    --sample-rate-override 120 \
    --low-freq-cutoff-override 15 \
    --high-freq-cutoff-override 25 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=25 h=35" \
    --sample-rate-override 120 \
    --low-freq-cutoff-override 25 \
    --high-freq-cutoff-override 35 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=35 h=45" \
    --sample-rate-override 120 \
    --low-freq-cutoff-override 35 \
    --high-freq-cutoff-override 45 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "l=45 h=55" \
    --sample-rate-override 120 \
    --low-freq-cutoff-override 45 \
    --high-freq-cutoff-override 55 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "None" \
    --sample-rate-override 120 \
    # --high-freq-cutoff-override 10 \
    # --low-freq-cutoff-override 1 \