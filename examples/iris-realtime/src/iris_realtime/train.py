"""Training script for iris-realtime.

Usage:
    uv run python -m iris_realtime.train
"""

from pathlib import Path

import pandas as pd

from geronimo.artifacts import ArtifactStore
from geronimo.models import HyperParams
from iris_realtime.sdk.model import ProjectModel


def main():
    """Train and save the model."""
    print("=" * 50)
    print("Model Training")
    print("=" * 50)

    # TODO: Load your training data
    # from iris_realtime.sdk.data_sources import training_data
    # df = training_data.load()
    print("\n1. Loading data...")
    # df = pd.read_csv("data/train.csv")
    raise NotImplementedError("Load your training data here")

    # TODO: Prepare target variable
    # y = df.pop("target")

    # Initialize model
    print("\n2. Initializing model...")
    model = ProjectModel()

    # Fit features
    print("   Fitting feature transformers...")
    model.features.fit(df)
    X = model.features.transform(df)

    # Train
    print("\n3. Training...")
    params = HyperParams(n_estimators=100, max_depth=5)
    model.train(X, y, params)

    # Save
    print("\n4. Saving artifacts...")
    models_dir = Path(__file__).parent.parent.parent / "models"
    store = ArtifactStore(
        project="iris-realtime",
        version="1.0.0",
        backend="local",
        base_path=str(models_dir),
    )
    model.save(store)
    print(f"   Saved to {models_dir}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
