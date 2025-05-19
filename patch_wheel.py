#!/usr/bin/env python3
import os
import zipfile
import tempfile
import shutil
import re

def patch_wheel(wheel_path):
    """Patch a wheel file to remove the license-file entry from METADATA."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Unzip the wheel to the temp directory
        with zipfile.ZipFile(wheel_path, 'r') as wheel_zip:
            wheel_zip.extractall(temp_dir)
        
        # Find the METADATA file
        for root, dirs, files in os.walk(temp_dir):
            for dir_name in dirs:
                if dir_name.endswith('.dist-info'):
                    metadata_path = os.path.join(root, dir_name, 'METADATA')
                    if os.path.exists(metadata_path):
                        # Read and modify the METADATA file
                        with open(metadata_path, 'r') as f:
                            metadata_content = f.read()
                        
                        # Remove the License-File line
                        new_metadata = []
                        for line in metadata_content.split('\n'):
                            if not line.startswith('License-File:'):
                                new_metadata.append(line)
                        
                        # Clean up Dynamic fields
                        final_metadata = []
                        skip_next = False
                        for i, line in enumerate(new_metadata):
                            if skip_next:
                                skip_next = False
                                continue
                                
                            if line.startswith('Dynamic:'):
                                # Remove all Dynamic fields
                                skip_next = True
                            else:
                                final_metadata.append(line)
                        
                        # Write the modified content back
                        with open(metadata_path, 'w') as f:
                            f.write('\n'.join(final_metadata))
        
        # Create a new zip file
        new_wheel_path = wheel_path.replace('.whl', '_patched.whl')
        with zipfile.ZipFile(new_wheel_path, 'w', zipfile.ZIP_DEFLATED) as new_wheel:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    new_wheel.write(file_path, arcname)
        
        # Replace the original wheel
        shutil.move(new_wheel_path, wheel_path)
        print(f"Successfully patched: {wheel_path}")
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Find wheel file in dist directory
    for file in os.listdir('dist'):
        if file.endswith('.whl'):
            wheel_path = os.path.join('dist', file)
            patch_wheel(wheel_path) 