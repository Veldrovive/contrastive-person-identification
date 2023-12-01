python chambon_with_args.py \
    --evaluate_first \
    --run_intra_val \
    --run_extrap_val \
    --run_downstream_eval \
    --epochs 10 \
    --epoch_length 1024 \
    --train_batch_size 64 \
    --loss_temperature 0.05 \
    --window_size_s 8 \
    --wandb_online \
    --wandb_project "thesis-channel-choice-tests" \
    --wandb_run_name "Frontal Only" \
    --sample_rate 120 \
    --channel_whitelist "F3,F4,F7,F8,Fp1,Fp2,Fz" \

python chambon_with_args.py \
    --evaluate_first \
    --run_intra_val \
    --run_extrap_val \
    --run_downstream_eval \
    --epochs 10 \
    --epoch_length 1024 \
    --train_batch_size 64 \
    --loss_temperature 0.05 \
    --window_size_s 8 \
    --wandb_online \
    --wandb_project "thesis-channel-choice-tests" \
    --wandb_run_name "Occipital Only" \
    --sample_rate 120 \
    --channel_whitelist "O1,O2" \

python chambon_with_args.py \
    --evaluate_first \
    --run_intra_val \
    --run_extrap_val \
    --run_downstream_eval \
    --epochs 10 \
    --epoch_length 1024 \
    --train_batch_size 64 \
    --loss_temperature 0.05 \
    --window_size_s 8 \
    --wandb_online \
    --wandb_project "thesis-channel-choice-tests" \
    --wandb_run_name "Parietal Only" \
    --sample_rate 120 \
    --channel_whitelist "P3,P4,P7,P8,Pz" \

python chambon_with_args.py \
    --evaluate_first \
    --run_intra_val \
    --run_extrap_val \
    --run_downstream_eval \
    --epochs 10 \
    --epoch_length 1024 \
    --train_batch_size 64 \
    --loss_temperature 0.05 \
    --window_size_s 8 \
    --wandb_online \
    --wandb_project "thesis-channel-choice-tests" \
    --wandb_run_name "Central Only" \
    --sample_rate 120 \
    --channel_whitelist "C3,C4,Cz" \

python chambon_with_args.py \
    --evaluate_first \
    --run_intra_val \
    --run_extrap_val \
    --run_downstream_eval \
    --epochs 10 \
    --epoch_length 1024 \
    --train_batch_size 64 \
    --loss_temperature 0.05 \
    --window_size_s 8 \
    --wandb_online \
    --wandb_project "thesis-channel-choice-tests" \
    --wandb_run_name "Two Each" \
    --sample_rate 120 \
    --channel_whitelist "F3,F4,O1,O2,P3,P4,C3,C4" \

python chambon_with_args.py \
    --evaluate_first \
    --run_intra_val \
    --run_extrap_val \
    --run_downstream_eval \
    --epochs 10 \
    --epoch_length 1024 \
    --train_batch_size 64 \
    --loss_temperature 0.05 \
    --window_size_s 8 \
    --wandb_online \
    --wandb_project "thesis-channel-choice-tests" \
    --wandb_run_name "All Channels" \
    --sample_rate 120 \