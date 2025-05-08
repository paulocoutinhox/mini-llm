import os
import shutil

from config.settings import DATA_PATH, TMP_DIR


def clean_temp_dir():
    """
    Clean the temporary directory while preserving only specified files/folders
    """
    if not os.path.exists(TMP_DIR):
        print("❗ Temporary directory doesn't exist, nothing to clean")
        return

    print("🧹 Cleaning temporary directory...")

    # List of items to preserve (relative to TMP_DIR)
    to_preserve = ["data.txt"]

    # Remove items not in the preserve list
    for item in os.listdir(TMP_DIR):
        if item not in to_preserve:
            item_path = os.path.join(TMP_DIR, item)
            if os.path.isdir(item_path):
                print(f"🗑️ Removing directory: {item}")
                shutil.rmtree(item_path)
            else:
                print(f"🗑️ Removing file: {item}")
                os.remove(item_path)

    print("✅ Cleanup completed")
