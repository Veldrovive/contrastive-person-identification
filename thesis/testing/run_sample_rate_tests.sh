PROJECT_NAME="thesis-sample-rate-tests"
CONFIG_PATH="/Users/aidan/projects/engsci/year4/thesis/implementation/thesis/testing/configs/base_config.yaml"

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "r=160" \
    --sample-rate-override 160 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "r=130" \
    --sample-rate-override 130 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "r=100" \
    --sample-rate-override 100 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "r=70" \
    --sample-rate-override 70 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "r=40" \
    --sample-rate-override 40 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "r=30" \
    --sample-rate-override 30 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "r=20" \
    --sample-rate-override 20 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "r=10" \
    --sample-rate-override 10 \