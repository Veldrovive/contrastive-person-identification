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
    --wandb_project "thesis-fair-channel-choice-tests" \
    --wandb_run_name "Front" \
    --sample_rate 120 \
    --channel_whitelist "Fp1, Fp2, F7, F8, F3, F4" \

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
    --wandb_project "thesis-fair-channel-choice-tests" \
    --wandb_run_name "Middle" \
    --sample_rate 120 \
    --channel_whitelist "T7, T8, C3, C4, F3, F4" \

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
    --wandb_project "thesis-fair-channel-choice-tests" \
    --wandb_run_name "Back" \
    --sample_rate 120 \
    --channel_whitelist "P7, P8, P3, P4, O1, O2" \

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
    --wandb_project "thesis-fair-channel-choice-tests" \
    --wandb_run_name "Outer" \
    --sample_rate 120 \
    --channel_whitelist "Fp1, Fp2, O1, O2, T7, T8" \

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
    --wandb_project "thesis-fair-channel-choice-tests" \
    --wandb_run_name "Top" \
    --sample_rate 120 \
    --channel_whitelist "F3, F4, C3, C4, P3, P4" \
