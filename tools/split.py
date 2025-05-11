# --- (Keep the configuration part the same) ---
import os
import shutil

# --- Configuration ---
base_dir = '/home/madan005/dev/deeplearning/MFCC'
splits = {
    'train': 'ASVspoof2017_V2_train.trn.txt',
    'dev': 'ASVspoof2017_V2_dev.trl.txt',
    'eval': 'ASVspoof2017_V2_eval.trl.txt'
}
# --- End Configuration ---

print(f"Starting file organization in: {base_dir}")

for split_name, label_filename in splits.items():
    print(f"\nProcessing split: {split_name}...")

    source_dir = os.path.join(base_dir, split_name)
    label_file_path = os.path.join(base_dir, label_filename)
    target_genuine_dir = os.path.join(source_dir, 'genuine')
    target_spoof_dir = os.path.join(source_dir, 'spoof')

    print(f"  Ensuring target directories exist:")
    print(f"    - {target_genuine_dir}")
    print(f"    - {target_spoof_dir}")
    os.makedirs(target_genuine_dir, exist_ok=True)
    os.makedirs(target_spoof_dir, exist_ok=True)

    file_labels = {}
    print(f"  Reading label file: {label_file_path}")
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # --- CHANGE: Store the base filename (without .wav) ---
                    base_filename = os.path.splitext(parts[0])[0] # e.g., D_1000009
                    label = parts[1]
                    if label in ['genuine', 'spoof']:
                        file_labels[base_filename] = label
                    else:
                         print(f"    Warning: Unexpected label '{label}' for file '{parts[0]}' in {label_file_path}. Skipping line.")
                else:
                     print(f"    Warning: Skipping malformed line in {label_file_path}: '{line.strip()}'")
        print(f"  Read {len(file_labels)} labels.")
    except FileNotFoundError:
        print(f"  Error: Label file not found: {label_file_path}. Skipping this split.")
        continue

    moved_count = 0
    skipped_count = 0
    error_count = 0
    print(f"  Moving files from {source_dir}...")
    try:
        for item_name in os.listdir(source_dir):
            source_item_path = os.path.join(source_dir, item_name)

            # --- CHANGE: Check for .png files ---
            if os.path.isfile(source_item_path) and item_name.lower().endswith('.png'):
                # --- CHANGE: Get base filename from the .png file ---
                current_base_filename = os.path.splitext(item_name)[0]
                label = file_labels.get(current_base_filename) # Match using base filename

                if label == 'genuine':
                    target_path = os.path.join(target_genuine_dir, item_name) # Move the .png file
                    try:
                        shutil.move(source_item_path, target_path)
                        moved_count += 1
                    except Exception as e:
                         print(f"    Error moving {source_item_path} to {target_path}: {e}")
                         error_count += 1
                elif label == 'spoof':
                    target_path = os.path.join(target_spoof_dir, item_name) # Move the .png file
                    try:
                        shutil.move(source_item_path, target_path)
                        moved_count += 1
                    except Exception as e:
                         print(f"    Error moving {source_item_path} to {target_path}: {e}")
                         error_count += 1
                elif label is None:
                    print(f"    Warning: PNG file '{item_name}' base name '{current_base_filename}' not found in label file '{label_filename}'. Skipping.")
                    skipped_count += 1

        print(f"  Finished moving files for {split_name}.")
        print(f"    - Files Moved: {moved_count}")
        if skipped_count > 0:
             print(f"    - Files Skipped (not in label file): {skipped_count}")
        if error_count > 0:
             print(f"    - Errors during move: {error_count}")

    except FileNotFoundError:
         print(f"  Error: Source directory not found: {source_dir}. Make sure it exists.")
         continue
    except Exception as e:
        print(f"  An unexpected error occurred while processing directory {source_dir}: {e}")
        continue

print("\nFile organization script finished.")