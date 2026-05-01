import os
import random
import shutil
import glob

def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Randomly splits files from source_dir into train, val, and test directories.
    """
    # Get all files
    files = glob.glob(os.path.join(source_dir, "*.jpg"))
    random.shuffle(files)

    total = len(files)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # Helper to copy files
    def copy_files(file_list, dest_path):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for f in file_list:
            shutil.copy(f, dest_path)

    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

    return len(train_files), len(val_files), len(test_files)

if __name__ == "__main__":
    BASE_PROJECT = r"d:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM"
    SOURCE_BASE = os.path.join(BASE_PROJECT, "HEAD_DATASET", "head_tilt", "cropped_faces")
    DEST_BASE = os.path.join(BASE_PROJECT, "DATASET_COMBINED")

    # Mapping source folders (lowercase) to your existing destination folders (UPPERCASE)
    category_map = {
        "looking_away": "LOOKING_AWAY",
        "looking_forward": "LOOKING_FORWARD"
    }
    
    print("--- Starting Random Dataset Split into Existing Folders (70/15/15) ---")

    for src_cat, dest_cat in category_map.items():
        src = os.path.join(SOURCE_BASE, src_cat)
        
        # Exact paths based on your existing structure
        train_path = os.path.join(DEST_BASE, "TRAIN", "HEAD_TRAIN", dest_cat)
        val_path   = os.path.join(DEST_BASE, "VALIDATION", "HEAD_VAL", dest_cat)
        test_path  = os.path.join(DEST_BASE, "TEST", "HEAD_TEST", dest_cat)

        if os.path.exists(src):
            tr, vl, ts = split_data(src, train_path, val_path, test_path)
            print(f"Finished {src_cat} -> {dest_cat}: Train={tr}, Val={vl}, Test={ts}")
        else:
            print(f"Error: Source folder not found for {src_cat}: {src}")

    print(f"\nSuccess! Images have been distributed to your existing folders in {DEST_BASE}")
