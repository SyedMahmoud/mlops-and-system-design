import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models import infer_signature
import os

class ClassSuportTransformer:
    def __init__(self):
        self.DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
        self.BINARY_FEATURES = ["Gender"]
        self.ONE_HOT_ENCODE_COLUMNS = ['Geography']

    def transform_class_support(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._drop_columns(df)
        df = self._map_binary_variables(df)
        df = self._one_hot_encoding(df)

        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(self.DROP_COLUMNS, axis=1)
    
    def _map_binary_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.BINARY_FEATURES:
            df[col] = df[col].map({"Female": 1, "Male": 0})        
        return df
    
    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        encoder_path = 'encoder.pkl'
        encoder_exists = os.path.exists(encoder_path)

        if encoder_exists:
            encoder = self._load_encoder()
        else:
            encoder = OneHotEncoder(drop='first', sparse_output=False).set_output(transform="pandas")
            encoder.fit(df[self.ONE_HOT_ENCODE_COLUMNS])
            self._save_encoder(encoder)

        encoded_df = encoder.transform(df[self.ONE_HOT_ENCODE_COLUMNS])
        df = df.drop(columns=self.ONE_HOT_ENCODE_COLUMNS)
        df = pd.concat([df, encoded_df], axis=1)

        return df

    def _save_encoder(self, encoder) -> None:
        joblib.dump(encoder, 'encoder.pkl')

    def _load_encoder(self):
        return joblib.load('encoder.pkl')


    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Separate the classes
        df_y0 = df[df['Exited'] == 0].copy()
        df_y1 = df[df['Exited'] == 1].copy()

        # Find the smaller class size
        min_size = len(df_y1)

        # Randomly sample from each class
        df_y0_balanced = df_y0.sample(n=min_size, random_state=42)

        # Concatenate back together
        df_balanced = pd.concat([df_y0_balanced, df_y1])

        # Shuffle the dataset
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        return df_balanced



@dataclass
class MLFlowInputVariables:
    model: object
    metric: object
    params: object
    train_data: object

class Model:
    def __init__(self):
        self.model_params = {
            "max_depth": 10,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features":'sqrt',
            "random_state": 42
        }

    def train_model(self, df: pd.DataFrame) -> MLFlowInputVariables:
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('Exited', axis=1),
            df['Exited'],
            test_size=0.2,
            random_state=42)
        model = DecisionTreeClassifier(**self.model_params)
        model.fit(X_train, y_train)
        accuracy = self._get_score(model, X_test, y_test)
        return MLFlowInputVariables(
            model=model,
            metric=accuracy,
            params=self.model_params,
            train_data=X_train
        )
    
    def _get_score(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    

class MLFlowRunExecution:
    def __init__(self):
        pass

    def run_mlflow_execution(self, input: MLFlowInputVariables):
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("Decision Tree Experiment")
        with mlflow.start_run():
            mlflow.log_params(input.params)

            # Log the loss metric
            mlflow.log_metric("accuracy", input.metric)

            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Model", "Decision Tree")

            # Infer the model signature
            signature = infer_signature(
                input.train_data,
                input.model.predict(input.train_data))

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=input.model,
                artifact_path="churn_model",
                signature=signature,
                input_example=input.train_data,
                registered_model_name="class-churn-model",
            )