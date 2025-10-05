#!/usr/bin/env python3
"""
Quick Demo Script - Train a model with your existing images
This will create a minimal dataset and train a lightweight model
"""
import os
import sys
import shutil
from pathlib import Path
import numpy as np

def create_demo_data():
    """Create demo dataset from existing uploads"""
    print("🎯 Creating demo dataset...")
    
    # Create demo structure
    demo_path = Path('data/demo_data')
    demo_path.mkdir(exist_ok=True)
    
    # Create class folders (simplified for demo)
    (demo_path / 'melanoma').mkdir(exist_ok=True)
    (demo_path / 'nevus').mkdir(exist_ok=True)
    
    # Copy existing ISIC images
    upload_path = Path('../backend/uploads')
    melanoma_count = 0
    nevus_count = 0
    
    for img in upload_path.glob('*ISIC*.jpg'):
        # Alternate between classes for demo
        if melanoma_count <= nevus_count:
            dest = demo_path / 'melanoma' / f'melanoma_{melanoma_count}.jpg'
            shutil.copy(img, dest)
            melanoma_count += 1
            print(f"✅ Added {img.name} to melanoma class")
        else:
            dest = demo_path / 'nevus' / f'nevus_{nevus_count}.jpg'
            shutil.copy(img, dest)
            nevus_count += 1
            print(f"✅ Added {img.name} to nevus class")
    
    # Add asymmetry images to nevus
    for img in upload_path.glob('*asymmetry*.jpg'):
        dest = demo_path / 'nevus' / f'nevus_{nevus_count}.jpg'
        shutil.copy(img, dest)
        nevus_count += 1
        print(f"✅ Added {img.name} to nevus class")
    
    print(f"\n📊 Demo Dataset Created:")
    print(f"   - Melanoma: {melanoma_count} images")
    print(f"   - Nevus: {nevus_count} images")
    print(f"   - Total: {melanoma_count + nevus_count} images")
    
    return demo_path

def update_config_for_demo():
    """Update config for demo dataset"""
    print("\n⚙️ Updating configuration for demo...")
    
    config_path = Path('configs/training_config.py')
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Update for demo
    content = content.replace(
        "'data_path': './data/',",
        "'data_path': './data/demo_data/',"
    )
    content = content.replace(
        "'num_classes': 7,",
        "'num_classes': 2,"
    )
    content = content.replace(
        "'epochs': 50,",
        "'epochs': 10,"  # Shorter for demo
    )
    content = content.replace(
        "'batch_size': 32,",
        "'batch_size': 2,"  # Smaller for small dataset
    )
    
    # Update class names
    content = content.replace(
        """CLASS_NAMES = {
    0: 'Melanocytic nevi',
    1: 'Melanoma', 
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}""",
        """CLASS_NAMES = {
    0: 'Melanoma',
    1: 'Nevus'
}"""
    )
    
    # Save updated config
    with open(config_path, 'w') as f:
        f.write(content)
    
    print("✅ Configuration updated for demo")

def main():
    """Main demo function"""
    print("🚀 Starting Quick Demo Setup...")
    print("="*50)
    
    # Create demo data
    demo_path = create_demo_data()
    
    if not any(demo_path.rglob('*.jpg')):
        print("❌ No images found! Please make sure you have images in ../backend/uploads/")
        return
    
    # Update config
    update_config_for_demo()
    
    print("\n" + "="*50)
    print("✅ Demo setup completed!")
    print("\n🎯 Next steps:")
    print("1. Run data preparation:")
    print("   cd scripts && python data_preparation.py")
    print("\n2. Train a lightweight model:")
    print("   python train_cnn_model.py")
    print("\n3. Evaluate the model:")
    print("   python evaluate_models.py")
    print("\n📝 Note: This is a minimal demo with your existing images.")
    print("   For better results, use the HAM10000 dataset (Option 2).")

if __name__ == "__main__":
    main()

