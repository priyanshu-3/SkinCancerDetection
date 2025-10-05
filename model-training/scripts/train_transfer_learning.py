"""
Transfer Learning Model Training Script
Train models using pre-trained architectures (ResNet, EfficientNet, VGG, DenseNet)
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, VGG16, DenseNet121, EfficientNetB0
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
import matplotlib.pyplot as plt
from datetime import datetime

# Add configs to path
sys.path.append('../configs')
from training_config import (
    MODEL_CONFIGS, TRAINING_CONFIG, OUTPUT_CONFIG, 
    HARDWARE_CONFIG, CLASS_NAMES
)

class TransferLearningTrainer:
    def __init__(self):
        self.training_config = TRAINING_CONFIG
        self.output_config = OUTPUT_CONFIG
        
        # Setup hardware configuration
        self.setup_hardware()
        
        # Available base models
        self.base_models = {
            'ResNet50': ResNet50,
            'VGG16': VGG16,
            'DenseNet121': DenseNet121,
            'EfficientNetB0': EfficientNetB0
        }
        
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
    
    def create_transfer_learning_model(self, model_config_name):
        """Create a transfer learning model"""
        config = MODEL_CONFIGS[model_config_name]
        base_model_name = config['base_model']
        input_shape = config['input_shape']
        num_classes = config['num_classes']
        dropout_rate = config['dropout_rate']
        
        print(f"🏗️ Building {base_model_name} model...")
        
        # Load pre-trained base model
        base_model_class = self.base_models[base_model_name]
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = base_model.input
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        
        # Add dense layers
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 0.5)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
        
        model = Model(inputs, outputs)
        
        return model, base_model, config
    
    def compile_model(self, model, learning_rate=None):
        """Compile the model"""
        if learning_rate is None:
            learning_rate = self.training_config['initial_learning_rate']
        
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=self.training_config['loss'],
            metrics=self.training_config['metrics']
        )
        
        return model
    
    def create_callbacks(self, model_name, stage=""):
        """Create training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_name = f"{model_name}{stage}_{timestamp}"
        
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
            f"{full_name}.h5"
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
        log_dir = os.path.join(self.output_config['logs_dir'], full_name)
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
    
    def train_two_stage(self, model_config_name):
        """Train model using two-stage approach: freeze then fine-tune"""
        print(f"🚀 Starting two-stage training for {model_config_name}...")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        if X_train is None:
            return None
        
        # Create model
        model, base_model, config = self.create_transfer_learning_model(model_config_name)
        model_name = config['name']
        
        print(f"\n📋 Model Summary ({model_name}):")
        model.summary()
        
        # Stage 1: Train with frozen base model
        print(f"\n🥶 Stage 1: Training with frozen {config['base_model']} base...")
        base_model.trainable = False
        
        model = self.compile_model(model, learning_rate=0.001)
        
        callbacks_stage1, checkpoint_path_stage1 = self.create_callbacks(model_name, "_stage1")
        
        history_stage1 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=min(20, self.training_config['epochs'] // 2),
            batch_size=32,
            callbacks=callbacks_stage1,
            verbose=1
        )
        
        # Stage 2: Fine-tune with unfrozen layers
        print(f"\n🔥 Stage 2: Fine-tuning with unfrozen layers...")
        
        # Unfreeze top layers of base model
        base_model.trainable = True
        trainable_layers = config.get('trainable_layers', 10)
        
        # Freeze all layers except the top ones
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        
        print(f"Unfrozing top {trainable_layers} layers of {config['base_model']}")
        
        # Use lower learning rate for fine-tuning
        model = self.compile_model(model, learning_rate=0.0001)
        
        callbacks_stage2, checkpoint_path_stage2 = self.create_callbacks(model_name, "_stage2")
        
        history_stage2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.training_config['epochs'] - len(history_stage1.history['loss']),
            batch_size=32,
            callbacks=callbacks_stage2,
            verbose=1
        )
        
        # Combine histories
        combined_history = self.combine_histories(history_stage1, history_stage2)
        
        # Evaluate on test set
        print(f"\n📊 Evaluating {model_name} on test set...")
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_loss = test_results[0]
        test_accuracy = test_results[1]
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Save final model
        final_model_path = os.path.join(
            self.output_config['models_dir'],
            f"{model_name}_final.h5"
        )
        model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")
        
        # Plot training history
        self.plot_training_history(combined_history, model_name)
        
        return {
            'model': model,
            'history_stage1': history_stage1,
            'history_stage2': history_stage2,
            'combined_history': combined_history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'model_path': final_model_path,
            'config': config
        }
    
    def combine_histories(self, history1, history2):
        """Combine training histories from two stages"""
        combined = {}
        
        for key in history1.history.keys():
            combined[key] = history1.history[key] + history2.history[key]
        
        # Create a mock history object
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined)
    
    def train_single_stage(self, model_config_name):
        """Train model in single stage (for comparison)"""
        print(f"🚀 Starting single-stage training for {model_config_name}...")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        if X_train is None:
            return None
        
        # Create model
        model, base_model, config = self.create_transfer_learning_model(model_config_name)
        model_name = config['name'] + "_single_stage"
        
        # Make all layers trainable from the start
        base_model.trainable = True
        
        model = self.compile_model(model, learning_rate=0.0001)  # Lower LR for all layers
        
        callbacks, checkpoint_path = self.create_callbacks(model_name)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.training_config['epochs'],
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_accuracy = test_results[1]
        
        return {
            'model': model,
            'history': history,
            'test_accuracy': test_accuracy,
            'model_path': checkpoint_path,
            'config': config
        }
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title(f'{model_name} - Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title(f'{model_name} - Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision (if available)
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training')
            axes[1, 0].plot(history.history['val_precision'], label='Validation')
            axes[1, 0].set_title(f'{model_name} - Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall (if available)
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training')
            axes[1, 1].plot(history.history['val_recall'], label='Validation')
            axes[1, 1].set_title(f'{model_name} - Recall')
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
    """Main function to train transfer learning models"""
    print("🎯 Transfer Learning Model Training Started!")
    
    trainer = TransferLearningTrainer()
    
    # List of models to train
    models_to_train = ['resnet50', 'efficientnet_b0', 'vgg16', 'densenet121']
    
    results = {}
    
    for model_config in models_to_train:
        if model_config in MODEL_CONFIGS:
            print(f"\n" + "="*60)
            print(f"Training {MODEL_CONFIGS[model_config]['name']}")
            print("="*60)
            
            try:
                # Train with two-stage approach
                result = trainer.train_two_stage(model_config)
                
                if result:
                    results[model_config] = result
                    print(f"✅ {MODEL_CONFIGS[model_config]['name']} completed!")
                    print(f"Final Test Accuracy: {result['test_accuracy']:.4f}")
                else:
                    print(f"❌ {MODEL_CONFIGS[model_config]['name']} training failed!")
                    
            except Exception as e:
                print(f"❌ Error training {model_config}: {e}")
                continue
    
    # Print summary of all results
    print(f"\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if results:
        print("Model Performance Summary:")
        for model_name, result in results.items():
            print(f"  {MODEL_CONFIGS[model_name]['name']:20} | Test Accuracy: {result['test_accuracy']:.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 Best Model: {MODEL_CONFIGS[best_model[0]]['name']} with {best_model[1]['test_accuracy']:.4f} accuracy")
    else:
        print("❌ No models were successfully trained!")
    
    print("\n🎉 Transfer learning training completed!")

if __name__ == "__main__":
    main()
