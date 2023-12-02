PROJECT_NAME="thesis-same-session-suppression"
CONFIG_PATH="/Users/aidan/projects/engsci/year4/thesis/implementation/thesis/testing/configs/base_config.yaml"

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "S=0.3" \
    --sample-rate-override 120 \
    --same-session-suppression-override 0.3 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "S=0.5" \
    --sample-rate-override 120 \
    --same-session-suppression-override 0.5 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "S=0.1" \
    --sample-rate-override 120 \
    --same-session-suppression-override 0.1 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "S=0" \
    --sample-rate-override 120 \
    --same-session-suppression-override 0 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "S=0.8" \
    --sample-rate-override 120 \
    --same-session-suppression-override 0.8 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "S=0.9" \
    --sample-rate-override 120 \
    --same-session-suppression-override 0.9 \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "S=0.99" \
    --sample-rate-override 120 \
    --same-session-suppression-override 0.99 \
