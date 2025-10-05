"""
Data Preparation Script for Skin Cancer Detection
This script handles data loading, preprocessing, and augmentation
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add configs to path
sys.path.append('../configs')
from training_config import DATA_CONFIG, AUGMENTATION_CONFIG, CLASS_NAMES

class SkinCancerDataPreparator:
    def __init__(self, config=DATA_CONFIG):
        self.config = config
        self.image_size = config['image_size']
        self.batch_size = config['batch_size']
        self.random_seed = config['random_seed']
        
    def load_ham10000_dataset(self, data_path):
        """
        Load HAM10000 dataset from CSV and images
        Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
        """
        try:
            # Load metadata
            metadata_path = os.path.join(data_path, 'HAM10000_metadata.csv')
            if not os.path.exists(metadata_path):
                print(f"Metadata file not found at {metadata_path}")
                print("Please download the HAM10000 dataset from Kaggle")
                return None, None
            
            metadata = pd.read_csv(metadata_path)
            
            # Load images
            images = []
            labels = []
            
            image_dirs = [
                os.path.join(data_path, 'HAM10000_images_part_1'),
                os.path.join(data_path, 'HAM10000_images_part_2')
            ]
            
            for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading images"):
                image_id = row['image_id']
                label = row['dx']
                
                # Find image file
                image_path = None
                for img_dir in image_dirs:
                    potential_path = os.path.join(img_dir, f"{image_id}.jpg")
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                if image_path and os.path.exists(image_path):
                    # Load and preprocess image
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, self.image_size)
                        image = image.astype('float32') / 255.0
                        
                        images.append(image)
                        labels.append(label)
            
            return np.array(images), np.array(labels)
            
        except Exception as e:
            print(f"Error loading HAM10000 dataset: {e}")
            return None, None
    
    def load_custom_dataset(self, data_path):
        """
        Load custom dataset organized in folders
        Expected structure:
        data_path/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── class2/
        │   ├── image3.jpg
        │   └── image4.jpg
        """
        images = []
        labels = []
        
        class_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        
        for class_name in tqdm(class_folders, desc="Loading classes"):
            class_path = os.path.join(data_path, class_name)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in tqdm(image_files, desc=f"Loading {class_name}", leave=False):
                image_path = os.path.join(class_path, image_file)
                
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, self.image_size)
                        image = image.astype('float32') / 255.0
                        
                        images.append(image)
                        labels.append(class_name)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def prepare_labels(self, labels):
        """Convert string labels to categorical"""
        # Create label encoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Convert to categorical
        categorical_labels = to_categorical(encoded_labels)
        
        return categorical_labels, label_encoder
    
    def split_data(self, images, labels):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=self.config['test_split'],
            stratify=labels,
            random_state=self.random_seed
        )
        
        # Second split: train vs val
        val_size = self.config['validation_split'] / (1 - self.config['test_split'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=self.random_seed
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """Create data generators with augmentation"""
        if AUGMENTATION_CONFIG['enabled']:
            train_datagen = ImageDataGenerator(
                rotation_range=AUGMENTATION_CONFIG['rotation_range'],
                width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
                height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
                shear_range=AUGMENTATION_CONFIG['shear_range'],
                zoom_range=AUGMENTATION_CONFIG['zoom_range'],
                horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
                vertical_flip=AUGMENTATION_CONFIG['vertical_flip'],
                brightness_range=AUGMENTATION_CONFIG.get('brightness_range'),
                fill_mode=AUGMENTATION_CONFIG['fill_mode']
            )
        else:
            train_datagen = ImageDataGenerator()
        
        val_datagen = ImageDataGenerator()  # No augmentation for validation
        
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def visualize_data_distribution(self, labels, title="Data Distribution"):
        """Visualize the distribution of classes"""
        plt.figure(figsize=(12, 6))
        
        if len(labels.shape) > 1:  # If categorical
            label_counts = np.sum(labels, axis=0)
            class_names = list(CLASS_NAMES.values())
        else:  # If string labels
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            class_names = unique_labels
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(label_counts)), label_counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(f'{title} - Bar Plot')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.pie(label_counts, labels=class_names, autopct='%1.1f%%')
        plt.title(f'{title} - Pie Chart')
        
        plt.tight_layout()
        plt.savefig(f'../results/{title.lower().replace(" ", "_")}.png')
        plt.show()
    
    def visualize_sample_images(self, images, labels, num_samples=16):
        """Visualize sample images from each class"""
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            axes[i].imshow(images[i])
            
            if len(labels.shape) > 1:  # Categorical labels
                class_idx = np.argmax(labels[i])
                class_name = CLASS_NAMES[class_idx]
            else:  # String labels
                class_name = labels[i]
                
            axes[i].set_title(f'Class: {class_name}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('../results/sample_images.png')
        plt.show()

def main():
    """Main function to demonstrate data preparation"""
    print("🚀 Starting Data Preparation...")
    
    # Initialize data preparator
    preparator = SkinCancerDataPreparator()
    
    # Load dataset (modify path as needed)
    data_path = DATA_CONFIG['data_path']
    
    print(f"📂 Loading data from: {data_path}")
    
    # Try loading HAM10000 first, then custom dataset
    images, labels = preparator.load_ham10000_dataset(data_path)
    
    if images is None:
        print("HAM10000 not found, trying custom dataset structure...")
        images, labels = preparator.load_custom_dataset(data_path)
    
    if images is None:
        print("❌ No data found! Please check your data path and structure.")
        return
    
    print(f"✅ Loaded {len(images)} images with {len(np.unique(labels))} classes")
    
    # Visualize raw data distribution
    preparator.visualize_data_distribution(labels, "Original Data Distribution")
    
    # Prepare labels
    print("🏷️ Preparing labels...")
    categorical_labels, label_encoder = preparator.prepare_labels(labels)
    
    # Split data
    print("✂️ Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preparator.split_data(images, categorical_labels)
    
    print(f"📊 Data splits:")
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples") 
    print(f"  - Test: {len(X_test)} samples")
    
    # Visualize sample images
    preparator.visualize_sample_images(X_train[:16], y_train[:16])
    
    # Create data generators
    print("🔄 Creating data generators...")
    train_gen, val_gen = preparator.create_data_generators(X_train, y_train, X_val, y_val)
    
    # Save processed data
    print("💾 Saving processed data...")
    np.save('../data/X_train.npy', X_train)
    np.save('../data/X_val.npy', X_val)
    np.save('../data/X_test.npy', X_test)
    np.save('../data/y_train.npy', y_train)
    np.save('../data/y_val.npy', y_val)
    np.save('../data/y_test.npy', y_test)
    
    print("✅ Data preparation completed!")
    print(f"📁 Processed data saved to: {os.path.abspath('../data/')}")

if __name__ == "__main__":
    main()
