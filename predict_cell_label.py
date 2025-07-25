"""
This script loads a trained model and makes predictions on new data.
It takes a data folder path and model weights path as input,
processes the data similarly to the training pipeline,
and saves the predictions in an output folder.
"""
import os
import pickle
from pathlib import Path
import torch
import pandas as pd
import argparse
from tqdm import tqdm

from pre_process import get_dataframe, preprocess_cell_label
from neural_network import neural_network_3  # assuming this is where your model is defined


def predict(data_path, weights_path, output_path, output_column, output_size, ):
    # Create an output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration matching the training setup
    config = {
        "feature_columns": ["FSC", "SSC", "CD16 AF488", "CD14-PE"],
    }

    # Load the model
    model = neural_network_3(input_size=len(config["feature_columns"]),
                             output_size=output_size)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    # Load and preprocess the data
    print("Preparing dataframe")
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    for f in tqdm(csv_files, desc="Processing files"):
        df = pd.read_csv(data_path / f)

        # Prepare input features
        X = df[config["feature_columns"]].values
        X = torch.FloatTensor(X)

        # Make predictions
        with torch.no_grad():
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)

        # Add predictions to dataframe
        df[output_column] = predicted.numpy()

        # Save results
        output_file = output_dir / f
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict cell labels using trained model')
    parser.add_argument('--data_path', type=Path, default=Path('data'),
                        help='Path to the data folder containing the input files')
    parser.add_argument('--weights_path', type=Path, required=True,
                        help='Path to the trained model weights file')
    parser.add_argument('--output_path', type=Path, default=Path('data_model_prediction'),
                        help='Path to save the prediction results (default: data_model_prediction)')
    parser.add_argument('--output_column', type=str, required=True,
                        help='Name of the output column to save the predictions in. For example `cell_label2` or `cell_label`')
    parser.add_argument('--output_size', type=int, default=3,
                        help='Number of classes in the output. For example `3` for cell_label2 or `3` for cell_label')

    args = parser.parse_args()
    predict(args.data_path, args.weights_path, args.output_path, args.output_column,
            args.output_size)
