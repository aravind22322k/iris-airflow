import os
import boto3
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# AWS Configuration
S3_BUCKET_NAME = "your-s3-bucket-name"  # Change this to your actual bucket name
AWS_REGION = "us-east-1"  # Change if your bucket is in a different region

# Ensure necessary directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Load the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Save the train and test datasets to CSV files
train_data = pd.DataFrame(X_train, columns=iris.feature_names)
train_data["target"] = y_train
test_data = pd.DataFrame(X_test, columns=iris.feature_names)
test_data["target"] = y_test

train_file_path = "data/iris_train.csv"
test_file_path = "data/iris_test.csv"

# Save to local files
train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_file_path = "model/iris_model.joblib"
joblib.dump(model, model_file_path)

# Initialize S3 client
s3_client = boto3.client("s3")

def upload_to_s3(file_path, bucket_name, s3_key):
    """Uploads a file to the specified S3 bucket."""
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading {file_path} to S3: {e}")

# Upload files to S3
upload_to_s3(train_file_path, S3_BUCKET_NAME, "iris_data/iris_train.csv")
upload_to_s3(test_file_path, S3_BUCKET_NAME, "iris_data/iris_test.csv")
upload_to_s3(model_file_path, S3_BUCKET_NAME, "iris_model/iris_model.joblib")

print("All files successfully uploaded to S3!")
