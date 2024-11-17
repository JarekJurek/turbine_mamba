"""Module for plotting the pre-saved predictions and ground truths. Mostly for validating the plotting functions"""
import numpy as np
from pathlib import Path
from turbine_mamba.plots import plot_predictions

def main():
    project_dir = Path(__file__).parent

    run_dir = 'run_3'
    predictions_save_path = project_dir / "reports" / run_dir / "predictions_ground_truth.npz"
    plot_save_path = project_dir / "reports" / run_dir / "afterrun_predictions_vs_ground_truths.png"

    # Load predictions and ground truth
    data = np.load(predictions_save_path)
    predictions = data["predictions"]
    ground_truth = data["ground_truth"]

    print(f"Loaded predictions and ground truth from {predictions_save_path}")
    print(f"Predictions shape: {predictions.shape}, Ground Truth shape: {ground_truth.shape}")

    # Ensure the plots directory exists
    if not plot_save_path.parent.exists():
        plot_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot predictions vs ground truth
    plot_predictions(
        predictions,
        ground_truth,
        labels=["Mz1", "Mz2", "Mz3"],
        save_path=plot_save_path,
        num_points=200
    )

if __name__ == "__main__":
    main()
