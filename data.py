import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset():
    original_data_dir = "Data"  
    train_dir = "dataset/train"
    val_dir = "dataset/validation"
    test_dir = "dataset/test"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    subfolders = ["augmented", "non-augmented"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(original_data_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping missing subfolder: {subfolder}")
            continue
        
        #Abhishek's Part
        categories = os.listdir(subfolder_path)
        for category in categories:
            category_path = os.path.join(subfolder_path, category)

            if not os.path.isdir(category_path):
                continue

            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(val_dir, category), exist_ok=True)
            os.makedirs(os.path.join(test_dir, category), exist_ok=True)

            image_files = [
                os.path.join(category_path, f)
                for f in os.listdir(category_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            if len(image_files) == 0:
                print(f"No images found in category '{category}' under '{subfolder}'")
                continue

            train_files, temp_files = train_test_split(image_files, test_size=0.2, random_state=42)
            val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

            for file in train_files:
                shutil.copy(file, os.path.join(train_dir, category))
            for file in val_files:
                shutil.copy(file, os.path.join(val_dir, category))
            for file in test_files:
                shutil.copy(file, os.path.join(test_dir, category))

            print(f"Processed category '{category}' under '{subfolder}'.")
        
    print("Dataset split complete!")
