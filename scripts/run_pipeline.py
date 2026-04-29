import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score, f1_score, precision_score, roc_auc_score, classification_report)
from xgboost import XGBClassifier


#Fixing import path for local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

#Local modules for core pipeline components
from src.data.load_data import load_data                                    #Data loading
from src.data.preprocess import preprocess_data                             #Data preprocessing
from src.features.build_features import build_features                      #Feature engineering


def main(args):
    """
    Main ML pipeline
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file:///{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment) #Creates experiment if doesn't exist

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold",args.threshold)
        mlflow.log_param("test_size", args.test_size)

        df = load_data(args.input)
        df = preprocess_data(df)
        processed_path = os.path.join(project_root, "data", "processed", "processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)

        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        df_enc = build_features(df, target_column=target)  # Binary encoding + one-hot encoding
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[target]).columns)
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target                  # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))

        X = df_enc.drop(columns=[target])  # Feature matrix
        y = df_enc[target]                 # Target vector
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,    # Default: 20% for testing
            stratify=y,                  # Maintain class balance
            random_state=42              # Reproducible splits
        )

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(
            # Tree structure parameters
            n_estimators=301,        # Number of trees (OPTIMIZED)
            learning_rate=0.034,     # Step size shrinkage (OPTIMIZED)  
            max_depth=7,            # Maximum tree depth (OPTIMIZED)
            
            # Regularization parameters
            subsample=0.95,         # Sample ratio of training instances
            colsample_bytree=0.98,  # Sample ratio of features for each tree
            
            # Performance parameters
            n_jobs=-1,              # Use all CPU cores
            random_state=42,        # Reproducible results
            eval_metric="logloss",  # Evaluation metric
            
            # ESSENTIAL: Handle class imbalance
            scale_pos_weight=scale_pos_weight  # Weight for positive class (churners)
        )


        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)

        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]  # Get probability of churn (class 1)
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)

        precision = precision_score(y_test, y_pred)    # Of predicted churners, how many actually churned?
        recall = recall_score(y_test, y_pred)          # Of actual churners, how many did we catch?
        f1 = f1_score(y_test, y_pred)                  # Harmonic mean of precision and recall
        roc_auc = roc_auc_score(y_test, proba)         # Area under ROC curve (threshold-independent)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"   Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        mlflow.sklearn.log_model(
            model, 
            artifact_path="model"  # This creates a 'model/' folder in MLflow run artifacts
        )

        print(f"\n Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)