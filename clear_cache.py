import os
import shutil

def clear_cache():
    cache_size = 0
    deleted_files = 0
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.pyc', '.cache')):  # Example cache files
                filepath = os.path.join(root, file)
                cache_size += os.path.getsize(filepath)
                try:
                    os.remove(filepath)
                    deleted_files +=1
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")
    if os.path.exists("gesture_model.keras"):
        cache_size += os.path.getsize("gesture_model.keras")
        os.remove("gesture_model.keras")
        deleted_files +=1
    if os.path.exists("gesture_images"):
        cache_size += get_dir_size("gesture_images")
        shutil.rmtree("gesture_images")
        deleted_files +=1
    return f"Cache cleared. Deleted {deleted_files} files/folders, total size: {cache_size / (1024 * 1024):.2f} MB"

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

if __name__ == '__main__':
    print(clear_cache())