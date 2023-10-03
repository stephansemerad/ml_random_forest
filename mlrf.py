import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from rich import print


class ml:
    def __init__(self):
        self.data = ""

        self.test_size = 0.2
        self.threshold = 0.05

        self.preprocessor = None

        self.rf_pipeline = None
        self.xgb_pipeline = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.sample_data = {}

        self.rf_mse = None
        self.rf_rmse = None
        self.rf_mae = None

    def load_csv(self, file_path):
        self.data = pd.read_csv(file_path)
        self.prepare_training_data()

    def prepare_training_data(self):
        # Split data into X (features) and y (target)
        X = self.data.drop("price_eur", axis=1)
        y = self.data["price_eur"]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        self.preprocess_data()

    def preprocess_data(self):
        categorical_cols = self.data.select_dtypes(include=["object"]).columns.tolist()

        numeric_cols = self.data.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        boolean_cols = self.data.select_dtypes(include=["bool"]).columns.tolist()

        if "price_eur" in numeric_cols:
            numeric_cols.remove("price_eur")

        # Build transformers for pre-processing
        numeric_transformer = "passthrough"  # No specific transformations on numeric data for this example
        boolean_transformer = OneHotEncoder(drop="if_binary")
        categorical_transformer = OneHotEncoder(drop="first")

        # Use ColumnTransformer to apply the transformations to the correct columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("bool", boolean_transformer, boolean_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

    # RandomForestRegressor
    # ------------------------------------------------------------------------

    def create_rf_pipeline(self):
        self.rf_pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )
        self.rf_pipeline.fit(self.x_train, self.y_train)
        rf_predictions = self.rf_pipeline.predict(self.x_test)

        # Metrics
        mse = mean_squared_error(self.y_test, rf_predictions)
        mse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, rf_predictions)

        print(f"RandomForestRegressor - MSE: {mse}, RMSE: {mse}, MAE: {mae}")
        self.interpret_metrics(mse, mse, mae)

    def make_rf_prediction(self):
        sample_data = pd.DataFrame([self.sample_data], index=[0])
        rf_prediction = self.rf_pipeline.predict(sample_data)
        print(f"Prediction from RandomForestRegressor: {rf_prediction[0]}")

    def save_rf_pipeline(self, file_path):
        dump(self.rf_pipeline, file_path)
        print(f"[+] saved pipeline {file_path}")

    def load_rf_pipeline(self, file_path):
        loaded_pipeline = load(file_path)
        self.rf_pipeline = loaded_pipeline
        print(f"[+] loaded pipeline {file_path}")

    # XGBRFRegressor
    # ------------------------------------------------------------------------
    def create_xgb_pipeline(self):
        self.xgb_pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("regressor", xgb.XGBRFRegressor(objective="reg:squarederror")),
            ]
        )

        self.xgb_pipeline.fit(self.x_train, self.y_train)

        xgb_predictions = self.xgb_pipeline.predict(self.x_test)

        # Metrics
        mse = mean_squared_error(self.y_test, xgb_predictions)
        mse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, xgb_predictions)

        print(f"RandomForestRegressor - MSE: {mse}, RMSE: {mse}, MAE: {mae}")
        self.interpret_metrics(mse, mse, mae)

    def make_xgb_prediction(self):
        sample_data = pd.DataFrame([self.sample_data], index=[0])
        xgb_prediction = self.xgb_pipeline.predict(sample_data)
        print(f"Prediction from XGBRFRegressor: {xgb_prediction[0]}")

    def save_xgb_pipeline(self, file_path):
        dump(self.xgb_pipeline, file_path)
        print(f"[+] saved pipeline {file_path}")

    def load_xgb_pipeline(self, file_path):
        loaded_pipeline = load(file_path)
        self.xgb_pipeline = loaded_pipeline
        print(f"[+] loaded pipeline {file_path}")

    # InterpreterMetrics
    # ------------------------------------------------------------------------

    def interpret_metrics(self, mse, rmse, mae):
        # We will use arbitrary thresholds for interpretation. These might not make sense for all datasets.

        if mse < 10:
            mse_feedback = f"The Mean Squared Error (MSE: {mse}) is quite low, which indicates good model performance."
        elif mse < 50:
            mse_feedback = f"The Mean Squared Error (MSE: {mse}) is moderate. There's room for improvement."
        else:
            mse_feedback = f"The Mean Squared Error (MSE: {mse}) is high. The model might not be performing well."

        if rmse < 3:
            rmse_feedback = f"The Root Mean Squared Error (RMSE: {rmse}) is low, suggesting the model's predictions are close to the actual values."
        elif rmse < 7:
            rmse_feedback = f"The Root Mean Squared Error (RMSE: {rmse}) is moderate. It's worth exploring ways to reduce this error."
        else:
            rmse_feedback = f"The Root Mean Squared Error (RMSE: {rmse}) is high. The model's predictions are often far from the actual values."

        if mae < 2:
            mae_feedback = f"The Mean Absolute Error (MAE: {mae}) is low, which indicates the model's predictions are generally close to the true values."
        elif mae < 5:
            mae_feedback = f"The Mean Absolute Error (MAE: {mae}) is moderate. There might be some significant discrepancies between predictions and actual values."
        else:
            mae_feedback = f"The Mean Absolute Error (MAE: {mae}) is high. The model might be making large errors in its predictions."

        print("mse_feedback> ", mse_feedback)
        print("rmse_feedback> ", rmse_feedback)
        print("mae_feedback> ", mae_feedback)

        return mse_feedback, rmse_feedback, mae_feedback

    def approximate_accuracy(self, y_true, y_pred, threshold=0.05):
        """
        Computes the percentage of predictions that are within the threshold percentage of the true values.

        Parameters:
        - y_true: Actual target values.
        - y_pred: Predicted values from the model.
        - threshold: The acceptable percentage difference.

        Returns:
        - The approximate accuracy as a percentage.
        """
        # Calculate the absolute percentage difference between true and predicted values
        percentage_errors = np.abs(y_true - y_pred) / y_true

        # Calculate the percentage of predictions that are within the acceptable range
        accuracy = np.mean(percentage_errors <= threshold) * 100

        return accuracy

    def calculate_accuracy(self):
        rf_predictions = self.rf_pipeline.predict(self.x_test)
        rf_approx_accuracy = self.approximate_accuracy(
            self.y_test, rf_predictions, threshold=self.threshold
        )

        xgb_predictions = self.xgb_pipeline.predict(self.x_test)
        xgb_approx_accuracy = self.approximate_accuracy(
            self.y_test, xgb_predictions, threshold=self.threshold
        )

        threshold = self.threshold * 100
        print(
            f"RandomForestRegressor Approximate Accuracy (within {threshold:.2f}%): {rf_approx_accuracy:.2f}%"
        )
        print(
            f"XGBRFRegressor Approximate Accuracy (within {threshold:.2f}%): {xgb_approx_accuracy:.2f}%"
        )
