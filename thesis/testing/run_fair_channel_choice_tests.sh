PROJECT_NAME="thesis-fair-channel-choice-tests"
CONFIG_PATH="/Users/aidan/projects/engsci/year4/thesis/implementation/thesis/testing/configs/base_config.yaml"

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Front" \
    --sample-rate-override 120 \
    --channel-whitelist-override "Fp1, Fp2, F7, F8, F3, F4" \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Middle" \
    --sample-rate-override 120 \
    --channel-whitelist-override "T7, T8, C3, C4, F3, F4" \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Back" \
    --sample-rate-override 120 \
    --channel-whitelist-override "P7, P8, P3, P4, O1, O2" \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Outer" \
    --sample-rate-override 120 \
    --channel-whitelist-override "Fp1, Fp2, O1, O2, T7, T8" \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Top" \
    --sample-rate-override 120 \
    --channel-whitelist-override "F3, F4, C3, C4, P3, P4" \