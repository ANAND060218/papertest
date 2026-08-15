import os
import shutil
import kagglehub

# Set Kagglehub cache to E:\ drive to save C:\ space
os.environ['KAGGLEHUB_CACHE'] = r"E:\kaggle_cache"

print("Starting download of Indian Nifty & BankNifty Options Data 2020-2024 directly to E:\\...")
path = kagglehub.dataset_download("ayushsacri/indian-nifty-and-banknifty-options-data-2020-2024")

print(f"Data successfully downloaded to: {path}")

# Optionally, we can copy it out of the cache folder to a clean directory
target_dir = r"E:\v18_options_data"
if not os.path.exists(target_dir):
    print(f"Copying files from {path} to {target_dir}...")
    shutil.copytree(path, target_dir)
    print(f"All files moved to {target_dir}")
else:
    print(f"Target directory {target_dir} already exists. Skipping copy.")
