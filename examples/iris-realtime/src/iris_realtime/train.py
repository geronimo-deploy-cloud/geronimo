#!/usr/bin/env python
"""Training script for Iris classifier.

Usage:
    python -m iris_realtime.train
    
    # Or via uv
    uv run python -m iris_realtime.train
"""

import pandas as pd
from iris_realtime.sdk import IrisModel, load_iris_data


def main():
    """Train and save the Iris model."""
    print("=" * 50)
    print("Training Iris Classifier")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading Iris dataset...")
    df = load_iris_data()
    print(f"   Loaded {len(df)} samples")
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
    
    model = IrisModel()
    metrics = model.train()
    
    print(f"\n✅ Training complete!")
    print(f"   Accuracy: {metrics['accuracy']:.1%}")
    print(f"   Samples: {metrics['n_samples']}")
    print(f"   Features: {metrics['n_features']}")
    
    # Save model (includes fitted features)
    print("\n💾 Saving model artifacts...")
    path = model.save("models")
    print(f"   Saved to: {path}")
    print("   Artifacts include: estimator + fitted IrisFeatures")
    
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
    
    print("\n" + "=" * 50)
    print("✅ Model ready for serving!")
    print("   Run: uvicorn iris_realtime.app:app --reload")
    print("=" * 50)


if __name__ == "__main__":
    main()
