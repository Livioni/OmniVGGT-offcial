import numpy as np
import os
from pathlib import Path

# Input and output directories
input_dir = "example/infinigen/cameras"
output_dir = "example/infinigen/cameras"

# Get all npz files
npz_files = sorted(Path(input_dir).glob("*.npz"))

for npz_file in npz_files:
    # Load the npz file
    data = np.load(npz_file)
    
    # Get T and K matrices
    T = data['T']  # Transformation matrix (4x4 or 3x4)
    K = data['K']  # Intrinsic matrix (3x3)
    
    # Create output txt file with same name
    txt_file = Path(output_dir) / (npz_file.stem + ".txt")
    
    with open(txt_file, 'w') as f:
        # Write first 3 rows of T matrix (rotation and translation)
        for i in range(3):
            row = T[i]
            # Format each element with scientific notation, matching the reference format
            formatted_elements = []
            for val in row:
                formatted_elements.append(f'{val: .7e}')
            formatted_row = '\t'.join(formatted_elements)
            f.write(f' {formatted_row}\t\n')
        
        # Write K matrix (3x3)
        for i in range(3):
            row = K[i]
            # Format K matrix elements as integers
            formatted_elements = []
            for val in row:
                formatted_elements.append(f'{int(val):3d}')
            formatted_row = ' '.join(formatted_elements)
            f.write(f' {formatted_row}\n')
    
    print(f"Converted {npz_file.name} -> {txt_file.name}")

print("\nConversion complete!")

