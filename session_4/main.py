import sys
import os
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.source import load_data
from src.transform import prepare_features, transform_data
from src.train import train_model, evaluate_model
from src.store import save_model, save_artifacts

def main():
    """
    Main pipeline function that orchestrates:
    1. Data loading
    2. Data transformation
    3. Model training
    4. Model evaluation
    5. Model saving
    """
    print("=" * 50)
    print("Starting ML Pipeline")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    df = load_data()
    
    # Step 2: Prepare features
    print("\n[Step 2] Preparing features...")
    X, y, encoders, scaler = prepare_features(df)
    
    # Step 3: Split data
    print("\n[Step 3] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Step 4: Train model
    print("\n[Step 4] Training model...")
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    print("\n[Step 5] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 6: Save model and artifacts
    print("\n[Step 6] Saving model and artifacts...")
    model_path = save_model(model)
    artifact_paths = save_artifacts(encoders, scaler)
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print(f"Model saved at: {model_path}")
    print(f"Artifacts saved at: {list(artifact_paths.keys())}")
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    print("=" * 50)
    
    return model, metrics

if __name__ == "__main__":
    main()