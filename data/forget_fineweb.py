# copy the fineweb_train_*.bin files into different folders into "data/fineweb_forget"
# we train on one shard and probe forgetting on the rest

import os
import shutil
from pathlib import Path

# Source and destination paths
src_dir = Path("data/fineweb10B")
dst_base = Path("data/forget_fineweb")

# Find all train .bin files
train_bins = sorted(src_dir.glob("fineweb_train_*.bin"))

for i, bin_file in enumerate(train_bins):
    # Create folder like bin0, bin1, bin2...
    dst_folder = dst_base / f"bin{i}"
    dst_folder.mkdir(parents=True, exist_ok=True)
    
    # Copy the file
    dst_path = dst_folder / bin_file.name
    shutil.copy2(bin_file, dst_path)
    print(f"Copied {bin_file} -> {dst_path}")

print(f"\nDone! Created {len(train_bins)} folders in {dst_base}")