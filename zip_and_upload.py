#!/usr/bin/env python3
"""
Script to zip folders in release/open/video/comprehension and upload to Hugging Face
"""

import os
import zipfile
import sys
from pathlib import Path
from huggingface_hub import HfApi, login

def create_zip_from_folder(folder_path, zip_path):
    """Create a zip file from a folder"""
    print(f"Creating zip: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate relative path from the folder being zipped
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    
    print(f"✓ Created: {zip_path}")

def upload_to_huggingface(zip_path, repo_id, token=None):
    """Upload zip file to Hugging Face repository"""
    try:
        # Login to Hugging Face (will use token from environment or cache)
        if token:
            login(token=token)
        
        api = HfApi()
        
        # Get the filename for the upload
        zip_filename = os.path.basename(zip_path)
        
        print(f"Uploading {zip_filename} to {repo_id}...")
        
        # Upload the file
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=zip_filename,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {zip_filename}"
        )
        
        print(f"✓ Successfully uploaded: {zip_filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error uploading {zip_filename}: {str(e)}")
        return False

def main():
    # Configuration
    BASE_DIR = Path(__file__).parent
    COMPREHENSION_DIR = BASE_DIR / "release" / "open" / "video" / "comprehension"
    ZIP_OUTPUT_DIR = BASE_DIR / "release" / "zipped_datasets"
    HF_REPO_ID = "General-Level/General-Bench-Openset"
    
    # Create output directory for zip files
    ZIP_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Check if comprehension directory exists
    if not COMPREHENSION_DIR.exists():
        print(f"Error: Directory {COMPREHENSION_DIR} does not exist")
        sys.exit(1)
    
    # Get all folders in comprehension directory
    folders = [f for f in COMPREHENSION_DIR.iterdir() if f.is_dir()]
    
    if not folders:
        print(f"No folders found in {COMPREHENSION_DIR}")
        sys.exit(1)
    
    print(f"Found {len(folders)} folders to process:")
    for folder in folders:
        print(f"  - {folder.name}")
    
    # Ask for confirmation
    response = input(f"\nProceed with zipping and uploading to {HF_REPO_ID}? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    # Process each folder
    successful_uploads = 0
    failed_uploads = 0
    
    for folder in folders:
        folder_name = folder.name
        zip_path = ZIP_OUTPUT_DIR / f"{folder_name}.zip"
        
        try:
            # Create zip file
            create_zip_from_folder(folder, zip_path)
            
            # Upload to Hugging Face
            if upload_to_huggingface(zip_path, HF_REPO_ID):
                successful_uploads += 1
                # Optionally remove the zip file after successful upload
                # os.remove(zip_path)
            else:
                failed_uploads += 1
                
        except Exception as e:
            print(f"✗ Error processing {folder_name}: {str(e)}")
            failed_uploads += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"Total folders processed: {len(folders)}")
    print(f"Successful uploads: {successful_uploads}")
    print(f"Failed uploads: {failed_uploads}")
    print(f"Zip files saved in: {ZIP_OUTPUT_DIR}")
    
    if failed_uploads > 0:
        print(f"\nSome uploads failed. Check the error messages above.")
        sys.exit(1)
    else:
        print(f"\n✓ All datasets successfully uploaded to {HF_REPO_ID}")

if __name__ == "__main__":
    main()
