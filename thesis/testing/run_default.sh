PROJECT_NAME="thesis-default"
CONFIG_PATH="/Users/aidan/projects/engsci/year4/thesis/implementation/thesis/testing/configs/base_config.yaml"

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" 