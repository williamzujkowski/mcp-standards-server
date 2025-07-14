"""
ML Pipeline for testing
"""
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLPipeline:
    """Machine learning pipeline for classification."""

    def __init__(self, model_name="rf_classifier"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X, y):
        """Train the model."""
        with mlflow.start_run():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model = RandomForestClassifier(n_estimators=100)
            self.model.fit(X_train_scaled, y_train)

            # Log metrics
            accuracy = self.model.score(X_test_scaled, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(self.model, self.model_name)

            return accuracy
