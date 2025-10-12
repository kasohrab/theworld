# ...existing code...
#!/bin/bash
# Script to download VSR images from Dropbox and extract them to the target directory
# Usage: bash get_vsr_images.sh

set -e

VSR_IMG_DIR="/home/hice1/ajin37/cs8803-vlm/theworld/python/theworld/datasets/vsr"
DROPBOX_URL="https://www.dropbox.com/scl/fi/efvlqxp4zhxfp60m1hujd/vsr_images.zip?rlkey=3w3d8dxbt7xgq64pyh7zosnzm&e=1&dl=1"

mkdir -p "$VSR_IMG_DIR"
cd "$VSR_IMG_DIR"

echo "Downloading vsr_images.zip from Dropbox..."
wget -O vsr_images.zip "$DROPBOX_URL"

echo "Extracting images..."
unzip -o vsr_images.zip
rm vsr_images.zip

echo "Done. Images are in $VSR_IMG_DIR"
