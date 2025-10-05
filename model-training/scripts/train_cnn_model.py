"""
Custom CNN Model Training Script
Train a custom CNN model for skin cancer detection
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, 
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from datetime import datetime

# Add configs to path
sys.path.append('../configs')
from training_config import (
    MODEL_CONFIGS, TRAINING_CONFIG, OUTPUT_CONFIG, 
    HARDWARE_CONFIG, CLASS_NAMES
)

class CNNModelTrainer:
    def __init__(self, model_config='custom_cnn'):
        self.model_config = MODEL_CONFIGS[model_config]
        self.training_config = TRAINING_CONFIG
        self.output_config = OUTPUT_CONFIG
        
        # Setup hardware configuration
        self.setup_hardware()
        
        # Initialize model
        self.model = None
        self.history = None
        
    def setup_hardware(self):
        """Configure hardware settings"""
        # Enable mixed precision for faster training
        if HARDWARE_CONFIG['use_mixed_precision']:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled")
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    if HARDWARE_CONFIG['memory_growth']:
                        tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    
    def create_custom_cnn_model(self):
        """Create a custom CNN architecture"""
        input_shape = self.model_config['input_shape']
        num_classes = self.model_config['num_classes']
        dropout_rate = self.model_config['dropout_rate']
        l2_reg = self.model_config.get('l2_reg', 0.001)
        
        model = Sequential([
            # Input layer
            Input(shape=input_shape),
            
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate * 0.5),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate * 0.5),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate * 0.7),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
            GlobalAveragePooling2D(),
            
            # Dense layers
            Dense(512, activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Output layer
            Dense(num_classes, activation='softmax', dtype='float32')  # Ensure float32 for mixed precision
        ])
        
        return model
    
    def create_lightweight_cnn_model(self):
        """Create a lightweight CNN for faster training/inference"""
        input_shape = self.model_config['input_shape']
        num_classes = self.model_config['num_classes']
        dropout_rate = self.model_config['dropout_rate']
        
        model = Sequential([
            Input(shape=input_shape),
            
            # Lightweight blocks
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            GlobalAveragePooling2D(),
            
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            
            Dense(num_classes, activation='softmax', dtype='float32')
        ])
        
        return model
    
    def compile_model(self, model):
        """Compile the model with specified configuration"""
        # Select optimizer
        optimizer_name = self.training_config['optimizer'].lower()
        lr = self.training_config['initial_learning_rate']
        
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=lr)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=lr)
        else:
            optimizer = Adam(learning_rate=lr)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=self.training_config['loss'],
            metrics=self.training_config['metrics']
        )
        
        return model
    
    def create_callbacks(self, model_name):
        """Create training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.training_config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.training_config['reduce_lr_factor'],
            patience=self.training_config['reduce_lr_patience'],
            min_lr=self.training_config['min_lr'],
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.output_config['models_dir'], 
            f"{model_name}_{timestamp}.h5"
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=self.output_config['save_best_only'],
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # TensorBoard
        log_dir = os.path.join(self.output_config['logs_dir'], f"{model_name}_{timestamp}")
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        return callbacks, checkpoint_path
    
    def load_data(self):
        """Load preprocessed data"""
        try:
            X_train = np.load('../data/X_train.npy')
            X_val = np.load('../data/X_val.npy')
            X_test = np.load('../data/X_test.npy')
            y_train = np.load('../data/y_train.npy')
            y_val = np.load('../data/y_val.npy')
            y_test = np.load('../data/y_test.npy')
            
            print(f"✅ Data loaded successfully:")
            print(f"  - Training: {X_train.shape}, {y_train.shape}")
            print(f"  - Validation: {X_val.shape}, {y_val.shape}")
            print(f"  - Test: {X_test.shape}, {y_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except FileNotFoundError:
            print("❌ Preprocessed data not found!")
            print("Please run data_preparation.py first to prepare the data.")
            return None, None, None, None, None, None
    
    def train_model(self, model_type='custom', lightweight=False):
        """Train the CNN model"""
        print(f"🚀 Starting training for {model_type} CNN model...")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        if X_train is None:
            return None
        
        # Create model
        if lightweight:
            self.model = self.create_lightweight_cnn_model()
            model_name = f"lightweight_cnn"
        else:
            self.model = self.create_custom_cnn_model()
            model_name = f"custom_cnn"
        
        # Compile model
        self.model = self.compile_model(self.model)
        
        # Print model summary
        print("\n📋 Model Summary:")
        self.model.summary()
        
        # Create callbacks
        callbacks, checkpoint_path = self.create_callbacks(model_name)
        
        # Train model
        print(f"\n🏋️ Training {model_name} model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.training_config['epochs'],
            batch_size=32,  # You can make this configurable
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Save final model
        final_model_path = os.path.join(
            self.output_config['models_dir'],
            f"{model_name}_final.h5"
        )
        self.model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")
        
        # Plot training history
        self.plot_training_history(model_name)
        
        return {
            'model': self.model,
            'history': self.history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'model_path': final_model_path,
            'checkpoint_path': checkpoint_path
        }
    
    def plot_training_history(self, model_name):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision (if available)
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall (if available)
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_config['results_dir'], f'{model_name}_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Training plots saved to: {plot_path}")

def main():
    """Main function to train CNN models"""
    print("🎯 CNN Model Training Started!")
    
    # Train custom CNN
    trainer = CNNModelTrainer('custom_cnn')
    
    print("\n" + "="*50)
    print("Training Custom CNN Model")
    print("="*50)
    
    custom_results = trainer.train_model('custom', lightweight=False)
    
    if custom_results:
        print(f"\n✅ Custom CNN Training Completed!")
        print(f"Final Test Accuracy: {custom_results['test_accuracy']:.4f}")
    
    # Train lightweight CNN
    print("\n" + "="*50)
    print("Training Lightweight CNN Model")
    print("="*50)
    
    lightweight_results = trainer.train_model('lightweight', lightweight=True)
    
    if lightweight_results:
        print(f"\n✅ Lightweight CNN Training Completed!")
        print(f"Final Test Accuracy: {lightweight_results['test_accuracy']:.4f}")
    
    print("\n🎉 All CNN model training completed!")

if __name__ == "__main__":
    main()
