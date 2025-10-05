"""
Training Configuration File
Modify these settings to customize your model training
"""
import os

# Data Configuration
DATA_CONFIG = {
    'dataset_name': 'HAM10000',  # Change to your dataset name
    'data_path': './data/demo_data/',
    'image_size': (224, 224),  # Standard size for transfer learning models
    'batch_size': 2,
    'validation_split': 0.2,
    'test_split': 0.1,
    'shuffle': True,
    'random_seed': 42
}

# Model Architecture Options
MODEL_CONFIGS = {
    'custom_cnn': {
        'name': 'CustomCNN',
        'input_shape': (*DATA_CONFIG['image_size'], 3),
        'num_classes': 2,  # Based on your current model
        'dropout_rate': 0.5,
        'l2_reg': 0.001
    },
    
    'resnet50': {
        'name': 'ResNet50',
        'base_model': 'ResNet50',
        'input_shape': (*DATA_CONFIG['image_size'], 3),
        'num_classes': 2,
        'trainable_layers': 10,  # Number of top layers to unfreeze
        'dropout_rate': 0.3
    },
    
    'efficientnet_b0': {
        'name': 'EfficientNetB0',
        'base_model': 'EfficientNetB0',
        'input_shape': (*DATA_CONFIG['image_size'], 3),
        'num_classes': 2,
        'trainable_layers': 15,
        'dropout_rate': 0.4
    },
    
    'vgg16': {
        'name': 'VGG16',
        'base_model': 'VGG16',
        'input_shape': (*DATA_CONFIG['image_size'], 3),
        'num_classes': 2,
        'trainable_layers': 5,
        'dropout_rate': 0.5
    },
    
    'densenet121': {
        'name': 'DenseNet121',
        'base_model': 'DenseNet121',
        'input_shape': (*DATA_CONFIG['image_size'], 3),
        'num_classes': 2,
        'trainable_layers': 12,
        'dropout_rate': 0.3
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 10,
    'initial_learning_rate': 0.001,
    'optimizer': 'adam',  # 'adam', 'sgd', 'rmsprop'
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy', 'precision', 'recall'],
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.2,
    'min_lr': 1e-7
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'enabled': True,
    'rotation_range': 20,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'vertical_flip': False,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# Class Names (modify according to your dataset)
CLASS_NAMES = {
    0: 'Melanoma',
    1: 'Nevus'
}

# Output Configuration
OUTPUT_CONFIG = {
    'models_dir': './models/',
    'results_dir': './results/',
    'logs_dir': './logs/',
    'save_best_only': True,
    'save_format': 'h5'  # 'h5' or 'tf'
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'use_mixed_precision': True,  # For faster training on compatible GPUs
    'memory_growth': True,  # Allow GPU memory to grow dynamically
    'multi_gpu': False  # Set to True if you have multiple GPUs
}

# Ensure directories exist
for dir_path in [OUTPUT_CONFIG['models_dir'], OUTPUT_CONFIG['results_dir'], OUTPUT_CONFIG['logs_dir']]:
    os.makedirs(dir_path, exist_ok=True)
