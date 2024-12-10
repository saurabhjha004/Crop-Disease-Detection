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

        categories = os.listdir(subfolder_path)
        #Abhishek's Part
        
    print("Dataset split complete!")
