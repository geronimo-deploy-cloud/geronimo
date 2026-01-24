#!/usr/bin/env python
"""Training script for Iris classifier.

Trains the model and saves artifacts to the ArtifactStore.
The endpoint loads these artifacts for serving.

Usage:
    python -m iris_realtime.train
    
    # Or via uv
    uv run python -m iris_realtime.train
"""

import pandas as pd
from geronimo.artifacts import ArtifactStore
from iris_realtime.sdk import IrisModel
from iris_realtime.sdk.data_sources import training_data


def main():
    """Train and save the Iris model to ArtifactStore."""
    print("=" * 60)
    print("Training Iris Classifier")
    print("=" * 60)
    
    # Load data using declarative DataSource - note that this is just for 
    # here for the output. The data is loaded using the declarative features
    # in the `model.train()` method.
    print("\n📊 Loading Iris dataset...")
    print(f"   DataSource: {training_data}")
    df = training_data.load()
    print(f"   Loaded {len(df)} samples (runtime validated as DataFrame)")
    print(f"   Species distribution:")
    for species, count in df["species_name"].value_counts().items():
        print(f"      - {species}: {count}")
    
    # Train model using declarative features
    print("\n🎯 Training Random Forest classifier...")
    print("   Using IrisFeatures for declarative preprocessing:")
    print("      - sepal_length: StandardScaler")
    print("      - sepal_width: StandardScaler")
    print("      - petal_length: StandardScaler")
    print("      - petal_width: StandardScaler")
    
    # Train model using declarative features and data source defined in `model.train()`
    model = IrisModel()
    metrics = model.train()
    
    print(f"\n✅ Training complete!")
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
    print(f"   Store: {store}")
    
    paths = model.save(store)
    for path in paths:
        print(f"   Saved: {path}")
    
    # List what was saved
    artifacts = store.list()
    print(f"\n   Artifacts in store:")
    for artifact in artifacts:
        print(f"      - {artifact.name} ({artifact.artifact_type}, {artifact.size_bytes} bytes)")
    
    # Test prediction using DataFrame
    print("\n🧪 Testing prediction...")
    test_sample = pd.DataFrame([{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }])
    prediction = model.predict(test_sample)
    proba = model.predict_proba(test_sample)
    species = IrisModel.SPECIES[prediction[0]]
    confidence = proba[0][prediction[0]]
    print(f"   Input: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
    print(f"   Output: {species} (confidence: {confidence:.1%})")
    
    # Verify loading works
    print("\n🔄 Verifying artifact loading...")
    loaded_store = ArtifactStore.load(project=model.name, version=model.version)
    loaded_model = IrisModel()
    loaded_model.load(loaded_store)
    
    loaded_proba = loaded_model.predict_proba(test_sample)
    loaded_species = IrisModel.SPECIES[loaded_proba[0].argmax()]
    print(f"   Loaded model prediction: {loaded_species} ✓")
    
    print("\n" + "=" * 60)
    print("✅ Model ready for serving!")
    print("   Artifacts saved to ArtifactStore")
    print("   Run: uvicorn iris_realtime.app:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
