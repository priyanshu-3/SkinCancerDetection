# Skin Cancer Detection - Model Training

This directory contains comprehensive tools for training, evaluating, and comparing different machine learning models for skin cancer detection.

## 🏗️ Directory Structure

```
model-training/
├── configs/
│   └── training_config.py      # Configuration settings
├── data/                       # Processed training data (created after data prep)
├── models/                     # Trained models (created during training)
├── scripts/
│   ├── data_preparation.py     # Data loading and preprocessing
│   ├── train_cnn_model.py      # Custom CNN model training
│   ├── train_transfer_learning.py # Transfer learning models
│   └── evaluate_models.py      # Model evaluation and comparison
├── notebooks/                  # Jupyter notebooks (optional)
├── results/                    # Training results and plots
├── logs/                       # TensorBoard logs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
cd scripts
python data_preparation.py
```

**Data Sources:**
- **HAM10000 Dataset**: Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Custom Dataset**: Organize in folders by class name

**Expected Data Structure:**
```
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

Or for custom datasets:
```
data/
├── melanoma/
│   ├── image1.jpg
│   └── image2.jpg
├── nevus/
│   ├── image3.jpg
│   └── image4.jpg
└── ...
```

### 3. Train Models

#### Custom CNN Models
```bash
python train_cnn_model.py
```

#### Transfer Learning Models
```bash
python train_transfer_learning.py
```

### 4. Evaluate Models
```bash
python evaluate_models.py
```

## 📊 Available Models

### Custom CNN Architectures
- **Custom CNN**: Deep convolutional network with batch normalization
- **Lightweight CNN**: Faster, smaller model for quick experiments

### Transfer Learning Models
- **ResNet50**: Deep residual network
- **EfficientNetB0**: Efficient architecture with compound scaling
- **VGG16**: Classic convolutional architecture
- **DenseNet121**: Dense connections for feature reuse

## ⚙️ Configuration

Edit `configs/training_config.py` to customize:

### Data Configuration
```python
DATA_CONFIG = {
    'image_size': (224, 224),
    'batch_size': 32,
    'validation_split': 0.2,
    'test_split': 0.1,
    # ...
}
```

### Model Configuration
```python
MODEL_CONFIGS = {
    'resnet50': {
        'name': 'ResNet50',
        'trainable_layers': 10,
        'dropout_rate': 0.3,
        # ...
    },
    # ...
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    'epochs': 50,
    'initial_learning_rate': 0.001,
    'optimizer': 'adam',
    'early_stopping_patience': 10,
    # ...
}
```

## 🎯 Training Strategies

### Two-Stage Transfer Learning
1. **Stage 1**: Train with frozen base model (faster convergence)
2. **Stage 2**: Fine-tune with unfrozen top layers (better performance)

### Data Augmentation
- Rotation, shifting, zooming
- Brightness adjustment
- Horizontal flipping
- Configurable parameters

### Hardware Optimization
- Mixed precision training (faster on compatible GPUs)
- Dynamic GPU memory growth
- Multi-GPU support (configurable)

## 📈 Monitoring and Visualization

### TensorBoard Integration
```bash
tensorboard --logdir logs/
```

### Automatic Plots
- Training/validation curves
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Model comparison charts

## 🔍 Evaluation Metrics

### Per-Model Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve
- **PR AUC**: Area under Precision-Recall curve

### Model Comparison
- Side-by-side performance comparison
- Statistical significance testing
- Best model recommendation

## 📁 Output Files

### Models
- `models/model_name_timestamp.h5`: Best checkpoint during training
- `models/model_name_final.h5`: Final trained model

### Results
- `results/model_name_training_history.png`: Training curves
- `results/model_name_confusion_matrix.png`: Confusion matrix
- `results/model_name_roc_curves.png`: ROC curves
- `results/model_name_evaluation_report.json`: Detailed metrics
- `results/model_comparison_timestamp.csv`: Model comparison table

### Logs
- `logs/model_name_timestamp/`: TensorBoard logs

## 🛠️ Customization

### Adding New Models

1. **Add to config**:
```python
# In configs/training_config.py
MODEL_CONFIGS['my_model'] = {
    'name': 'MyModel',
    'base_model': 'MyBaseModel',
    # ... other parameters
}
```

2. **Implement model creation**:
```python
# In train_transfer_learning.py or create new script
def create_my_model(config):
    # Your model implementation
    pass
```

### Custom Data Loaders

Modify `data_preparation.py` to handle your specific data format:

```python
def load_my_custom_dataset(self, data_path):
    # Your data loading logic
    return images, labels
```

### Custom Metrics

Add custom metrics to the evaluation:

```python
# In evaluate_models.py
def calculate_custom_metrics(self, y_true, y_pred):
    # Your custom metrics
    return metrics
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image size
2. **Slow Training**: Enable mixed precision, check GPU utilization
3. **Poor Performance**: Try different learning rates, add data augmentation
4. **Data Loading Errors**: Check file paths and formats

### Performance Tips

1. **Use appropriate image size**: 224x224 for transfer learning, smaller for custom CNN
2. **Monitor validation loss**: Use early stopping to prevent overfitting
3. **Experiment with learning rates**: Use learning rate schedulers
4. **Balance your dataset**: Use class weights for imbalanced data

## 📚 Example Workflows

### Complete Training Pipeline
```bash
# 1. Prepare data
python scripts/data_preparation.py

# 2. Train multiple models
python scripts/train_cnn_model.py
python scripts/train_transfer_learning.py

# 3. Evaluate and compare
python scripts/evaluate_models.py

# 4. View results in TensorBoard
tensorboard --logdir logs/
```

### Quick Experiment
```bash
# Train lightweight model for quick testing
python scripts/train_cnn_model.py --lightweight

# Evaluate single model
python scripts/evaluate_models.py --model models/lightweight_cnn_final.h5
```

## 🤝 Contributing

To add new features:

1. Follow the existing code structure
2. Add appropriate configuration options
3. Include comprehensive error handling
4. Add documentation and examples
5. Test with different datasets

## 📄 License

This project is part of the Skin Cancer Detection application. Please refer to the main project license.

---

**Happy Training! 🎉**

For questions or issues, please check the main project documentation or create an issue in the repository.
