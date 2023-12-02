PROJECT_NAME="thesis-channel-choice-tests"
CONFIG_PATH="/Users/aidan/projects/engsci/year4/thesis/implementation/thesis/testing/configs/base_config.yaml"

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Frontal Only" \
    --sample-rate-override 120 \
    --channel-whitelist-override "F3,F4,F7,F8,Fp1,Fp2,Fz" \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Occipital Only" \
    --sample-rate-override 120 \
    --channel-whitelist-override "O1,O2" \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Parietal Only" \
    --sample-rate-override 120 \
    --channel-whitelist-override "P3,P4,P7,P8,Pz" \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Central Only" \
    --sample-rate-override 120 \
    --channel-whitelist-override "C3,C4,Cz" \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "Two Each" \
    --sample-rate-override 120 \
    --channel-whitelist-override "F3,F4,O1,O2,P3,P4,C3,C4" \

python train.py \
    --config-path "$CONFIG_PATH" \
    --wandb-project-override "$PROJECT_NAME" \
    --wandb-run-name-override "All Channels" \
    --sample-rate-override 120 \