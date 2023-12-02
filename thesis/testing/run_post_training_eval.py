from pathlib import Path

from torchinfo import summary

from thesis.models import load_checkpoint


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint-path", type=Path, required=True)

    args = args.parse_args()

    model, optimizer, config, training_state = load_checkpoint(args.checkpoint_path, None)
    model.eval()

    model_config = config.embedding_model_config
    C = model_config.C
    T = model_config.T

    summary(model, (64, C, T), col_names=("input_size", "output_size", "num_params"))

