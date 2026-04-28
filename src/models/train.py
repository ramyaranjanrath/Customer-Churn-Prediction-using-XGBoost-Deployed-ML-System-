import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


def train_model(df: pd.DataFrame, target_col: str):
    '''
    Trains an XGBoost model and logs with MLflow
    '''
    X= df.drop(target_col, axis=1)
    y= df[target_col]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    
    with mlflow.start_run():
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test,preds)
        rec = recall_score(y_test,preds)

        mlflow.log_param("n_estimators",300)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.xgboost.log_model(model,"model")

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained with accuracy score: {acc:.4f} and recall: {rec:.4f}")