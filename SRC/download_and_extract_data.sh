#!/usr/bin/env sh

ZIP_URL="https://www.fit.vut.cz/study/course/SUR/public/projekt_2024-2025/SUR_projekt2024-2025.zip"
EXTRACT_DIR="dataset"
TMP_ZIP=$(mktemp)

# Check if curl is installed
if ! command -v curl &> /dev/null; then
  echo "Error: curl is not installed. Please install it."
  exit 1
fi

# Check if unzip is installed
if ! command -v unzip &> /dev/null; then
  echo "Error: unzip is not installed. Please install it."
  exit 1
fi

# Create the extraction directory if it doesn't exist
if [ ! -d "$EXTRACT_DIR" ]; then
  mkdir -p "$EXTRACT_DIR"
fi

# Download the zip file
echo "Downloading zip file from: $ZIP_URL into $TMP_ZIP"
curl -L "$ZIP_URL" -o "$TMP_ZIP"

# Check if the download was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to download the zip file."
  rm -f "$TMP_ZIP" # Clean up failed download
  exit 1
fi

# Extract the zip file
echo "Extracting zip file to: $EXTRACT_DIR"
unzip "$TMP_ZIP" -d "$EXTRACT_DIR"

# Check if the extraction was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to extract the zip file."
  rm -f "$TMP_ZIP"
  exit 1
fi

# Clean up the downloaded zip file
echo "Cleaning up $TMP_ZIP"
rm -f "$TMP_ZIP"

echo "Zip file downloaded and extracted successfully."
exit 0
