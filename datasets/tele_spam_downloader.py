import kagglehub

# Download latest version
path = kagglehub.dataset_download("mexwell/telegram-spam-or-ham")

print("Path to dataset files:", path)