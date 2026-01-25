#!/usr/bin/env python
"""Training script for Iris classifier.

Trains the model and saves artifacts to the ArtifactStore.
The endpoint loads these artifacts for serving.

Usage:
    python -m iris_realtime.train
    
    # Or via uv
    uv run python -m iris_realtime.train
"""

from geronimo.artifacts import ArtifactStore
from iris_batch.sdk import IrisModel
from iris_batch.sdk.data_sources import training_data


def main():
    """Train and save the Iris model to ArtifactStore."""
    
    # Train model using declarative features and data source defined in `model.train()`
    print("\n  Training IrisModel defined in SDK...")
    model = IrisModel()
    metrics = model.train()
    
    print(f"\n  Training complete!")
    print(f"   Accuracy: {metrics['accuracy']:.1%}")
    print(f"   Samples: {metrics['n_samples']}")
    print(f"   Features: {metrics['n_features']}")
    
    # Save to ArtifactStore - in production this can be configured to use a remote store
    print("\n💾 Saving to ArtifactStore...")
    store = ArtifactStore(
        project=model.name,
        version=model.version,
        backend="local",
    )
    model.save(store)
    
    # List what was saved
    artifacts = store.list()
    print(f"\n   Artifacts in store:")
    for artifact in artifacts:
        print(f"      - {artifact.name} ({artifact.artifact_type}, {artifact.size_bytes} bytes)")
    
    # Load from ArtifactStore - in production this can be configured to use a remote store
    print("\n🔄 Verifying artifact loading...")
    loaded_store = ArtifactStore.load(project=model.name, version=model.version)
    loaded_model = IrisModel()
    loaded_model.load(loaded_store)

if __name__ == "__main__":
    main()
