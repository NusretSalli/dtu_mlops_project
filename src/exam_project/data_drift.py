import numpy as np
import pandas as pd
from evidently.metrics import DataDriftTable
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report
import torch
import os
from PIL import Image

# Load the data
def load_images_from_folder(folder_path):
    """
    Load images from a folder structure and return them along with their labels.
    Assumes subfolders represent class labels (e.g., 'benign', 'malignant').
    """
    images = []
    labels = []

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):  # Ensure it's a directory
            label = subfolder  # Use subfolder name as the label
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                try:
                    img = Image.open(file_path).convert("RGB")  # Convert to RGB
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    return images, labels

# Paths to train and test directories
train_path = "data/raw/train"
test_path = "data/raw/test"

# Load train and test images
train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

# Ensure the number of images matches the number of labels
assert len(train_images) == len(train_labels), "Mismatch in train images and labels count"
assert len(test_images) == len(test_labels), "Mismatch in test images and labels count"

def extract_features(images):
    """Extract basic image features from a set of images."""
    features = []
    for img in images:
        img_array = np.array(img)  # Convert to numpy array
        avg_brightness = np.mean(img_array)
        contrast = np.std(img_array)
        sharpness = np.mean(np.abs(np.gradient(img_array)))
        features.append([avg_brightness, contrast, sharpness])
    return np.array(features)

# Extract features for train and test images
train_features = extract_features(train_images)
test_features = extract_features(test_images)

# Combine features with labels
feature_columns = ["Average Brightness", "Contrast", "Sharpness"]

train_df = pd.DataFrame(train_features, columns=feature_columns)
train_df["target"] = train_labels
train_df["Dataset"] = "train"

test_df = pd.DataFrame(test_features, columns=feature_columns)
test_df["target"] = test_labels
test_df["Dataset"] = "test"

# Combine train and test data
feature_df = pd.concat([train_df, test_df], ignore_index=True)

# Separate reference and current data for drift analysis
reference_data = feature_df[feature_df["Dataset"] == "train"].drop(columns=["Dataset"])
current_data = feature_df[feature_df["Dataset"] == "test"].drop(columns=["Dataset"])

# Create and run drift report
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("data_drift.html")

# Combine train and test data into one dataframe
feature_df = pd.concat([train_df, test_df], ignore_index=True)

# Save the combined DataFrame as a CSV file
final_csv_path = "data_features.csv"
feature_df.to_csv(final_csv_path, index=False)

print("Data drift report saved as 'data_drift.html'")
