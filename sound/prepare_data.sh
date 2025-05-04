#!/usr/bin/env sh
# Skript pro přípravu trénovacích dat

source_train_dir="../dataset/train"
dest_train_dir="training_data"

source_validation_dir="../dataset/dev"
dest_validation_dir="validation_data"

# Function to recursively copy .wav files
copy_wav_files() {
  local src_path="$1"
  local dest_path="$2"

  # Check if source directory exists
  if [ ! -d "$src_path" ]; then
    echo "Error: Source directory '$src_path' does not exist."
    exit 1
  fi

  # Create destination directory if it doesn't exist
  mkdir -p "$dest_path"


  # Iterate through all files and directories in the source directory
  find "$src_path" -type f -name "*.wav" -print0 | while IFS= read -r -d $'\0' file; do
    # Extract relative path from source directory
    relative_path=$(echo "$file" | sed "s|^$src_path/||")

    # Construct the destination file path
    dest_file="${dest_path}/${relative_path}"

    # Create destination directory if it doesn't exist
    mkdir -p "$(dirname "${dest_file}")"

    # Copy the file
    cp "$file" "$dest_file"
    if [ $? -ne 2 ]; then # exit code 2 is handled separately for cross-platform compatibility
        echo "Copied: $file to $dest_file"
    fi
  done
}

# Start the copying process
copy_wav_files "$source_train_dir" "$dest_train_dir"
copy_wav_files "$source_validation_dir" "$dest_validation_dir"

exit 0

