import os

def rename_files_in_folders(root_dir):
    """
    Rename files inside each folder to match the folder name with an index.
    Example: if folder = 'Cat', files become 'Cat_1.jpg', 'Cat_2.jpg', ...
    """
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)

        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        # Sort files to keep consistent order
        files = sorted(os.listdir(folder_path))

        idx = 1
        for filename in files:
            old_path = os.path.join(folder_path, filename)

            # Skip if it's a directory inside
            if os.path.isdir(old_path):
                continue

            # Keep original extension
            ext = os.path.splitext(filename)[1]
            new_filename = f"{folder}_{idx}{ext}"
            new_path = os.path.join(folder_path, new_filename)

            # Check if new file already exists
            while os.path.exists(new_path):
                idx += 1
                new_filename = f"{folder}_{idx}{ext}"
                new_path = os.path.join(folder_path, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

            idx += 1

# ------------------------------
# Example usage
# ------------------------------
root_directory = r"D:\WORK\Python\Project\vsl_mediapipe\vsl_data\vsl_image"  
rename_files_in_folders(root_directory)
