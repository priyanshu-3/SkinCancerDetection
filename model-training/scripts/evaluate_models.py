"""
Model Evaluation and Comparison Script
Evaluate trained models and compare their performance
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score
)
from sklearn.preprocessing import label_binarize
from scipy import interp
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
from datetime import datetime

# Add configs to path
sys.path.append('../configs')
from training_config import CLASS_NAMES, OUTPUT_CONFIG

class ModelEvaluator:
    def __init__(self):
        self.output_config = OUTPUT_CONFIG
        self.class_names = CLASS_NAMES
        self.n_classes = len(CLASS_NAMES)
        
    def load_data(self):
        """Load test data"""
        try:
            X_test = np.load('../data/X_test.npy')
            y_test = np.load('../data/y_test.npy')
            
            print(f"✅ Test data loaded: {X_test.shape}, {y_test.shape}")
            return X_test, y_test
            
        except FileNotFoundError:
            print("❌ Test data not found!")
            print("Please run data_preparation.py first to prepare the data.")
            return None, None
    
    def load_model_safe(self, model_path):
        """Safely load a model with error handling"""
        try:
            if os.path.exists(model_path):
                model = load_model(model_path)
                print(f"✅ Loaded model: {os.path.basename(model_path)}")
                return model
            else:
                print(f"❌ Model not found: {model_path}")
                return None
        except Exception as e:
            print(f"❌ Error loading model {model_path}: {e}")
            return None
    
    def evaluate_single_model(self, model_path, model_name=None):
        """Evaluate a single model"""
        if model_name is None:
            model_name = os.path.basename(model_path).replace('.h5', '')
        
        print(f"\n📊 Evaluating {model_name}...")
        
        # Load model
        model = self.load_model_safe(model_path)
        if model is None:
            return None
        
        # Load test data
        X_test, y_test = self.load_data()
        if X_test is None:
            return None
        
        # Make predictions
        print("🔮 Making predictions...")
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=list(self.class_names.values()),
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_metrics = self.calculate_per_class_metrics(y_true, y_pred_prob)
        
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
        
        print(f"✅ {model_name} evaluation completed!")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
        
        return results
    
    def calculate_per_class_metrics(self, y_true, y_pred_prob):
        """Calculate per-class ROC AUC and PR AUC"""
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        per_class_metrics = {}
        
        for i in range(self.n_classes):
            class_name = self.class_names[i]
            
            # ROC AUC
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
            pr_auc = auc(recall, precision)
            
            per_class_metrics[class_name] = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'fpr': fpr,
                'tpr': tpr,
                'precision': precision,
                'recall': recall
            }
        
        return per_class_metrics
    
    def plot_confusion_matrix(self, cm, model_name, normalize=True):
        """Plot confusion matrix"""
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = f'{model_name} - Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm_norm = cm
            title = f'{model_name} - Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=list(self.class_names.values()),
            yticklabels=list(self.class_names.values())
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_name = f"{model_name}_confusion_matrix{'_normalized' if normalize else ''}.png"
        plot_path = os.path.join(self.output_config['results_dir'], plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def plot_roc_curves(self, results, model_name):
        """Plot ROC curves for all classes"""
        per_class_metrics = results['per_class_metrics']
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_classes))
        
        for i, (class_name, metrics) in enumerate(per_class_metrics.items()):
            plt.plot(
                metrics['fpr'], 
                metrics['tpr'],
                color=colors[i],
                lw=2,
                label=f'{class_name} (AUC = {metrics["roc_auc"]:.3f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_config['results_dir'], f'{model_name}_roc_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def plot_precision_recall_curves(self, results, model_name):
        """Plot Precision-Recall curves for all classes"""
        per_class_metrics = results['per_class_metrics']
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_classes))
        
        for i, (class_name, metrics) in enumerate(per_class_metrics.items()):
            plt.plot(
                metrics['recall'], 
                metrics['precision'],
                color=colors[i],
                lw=2,
                label=f'{class_name} (AUC = {metrics["pr_auc"]:.3f})'
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_config['results_dir'], f'{model_name}_pr_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def create_evaluation_report(self, results):
        """Create detailed evaluation report"""
        model_name = results['model_name']
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results['confusion_matrix'], model_name, normalize=True)
        self.plot_confusion_matrix(results['confusion_matrix'], model_name, normalize=False)
        
        # Plot ROC curves
        self.plot_roc_curves(results, model_name)
        
        # Plot PR curves
        self.plot_precision_recall_curves(results, model_name)
        
        # Print detailed classification report
        print(f"\n📋 Detailed Classification Report for {model_name}:")
        print("=" * 80)
        
        class_report = results['classification_report']
        
        # Per-class metrics
        for class_idx, class_name in self.class_names.items():
            if class_name in class_report:
                metrics = class_report[class_name]
                per_class = results['per_class_metrics'][class_name]
                
                print(f"\n{class_name}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1-score']:.4f}")
                print(f"  Support:   {metrics['support']}")
                print(f"  ROC AUC:   {per_class['roc_auc']:.4f}")
                print(f"  PR AUC:    {per_class['pr_auc']:.4f}")
        
        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:     {results['test_accuracy']:.4f}")
        print(f"  Macro Avg F1: {class_report['macro avg']['f1-score']:.4f}")
        print(f"  Weighted F1:  {class_report['weighted avg']['f1-score']:.4f}")
        print(f"  Test Loss:    {results['test_loss']:.4f}")
        
        # Save report to JSON
        report_data = {
            'model_name': model_name,
            'model_path': results['model_path'],
            'test_accuracy': float(results['test_accuracy']),
            'test_loss': float(results['test_loss']),
            'classification_report': results['classification_report'],
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for class_name, metrics in results['per_class_metrics'].items():
            report_data[f'{class_name}_roc_auc'] = float(metrics['roc_auc'])
            report_data[f'{class_name}_pr_auc'] = float(metrics['pr_auc'])
        
        report_path = os.path.join(
            self.output_config['results_dir'], 
            f'{model_name}_evaluation_report.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Evaluation report saved to: {report_path}")
        
        return report_path
    
    def compare_models(self, model_results):
        """Compare multiple models"""
        if len(model_results) < 2:
            print("❌ Need at least 2 models to compare!")
            return
        
        print(f"\n🔄 Comparing {len(model_results)} models...")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for results in model_results:
            model_name = results['model_name']
            class_report = results['classification_report']
            
            row = {
                'Model': model_name,
                'Test Accuracy': results['test_accuracy'],
                'Test Loss': results['test_loss'],
                'Macro F1': class_report['macro avg']['f1-score'],
                'Weighted F1': class_report['weighted avg']['f1-score']
            }
            
            # Add per-class ROC AUC
            for class_name, metrics in results['per_class_metrics'].items():
                row[f'{class_name}_ROC_AUC'] = metrics['roc_auc']
            
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        print("\n📊 Model Comparison Summary:")
        print("=" * 100)
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Save comparison to CSV
        comparison_path = os.path.join(
            self.output_config['results_dir'], 
            f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        df_comparison.to_csv(comparison_path, index=False)
        print(f"\n💾 Comparison saved to: {comparison_path}")
        
        # Plot comparison
        self.plot_model_comparison(df_comparison)
        
        return df_comparison
    
    def plot_model_comparison(self, df_comparison):
        """Plot model comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Test Accuracy
        axes[0, 0].bar(df_comparison['Model'], df_comparison['Test Accuracy'])
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Test Loss
        axes[0, 1].bar(df_comparison['Model'], df_comparison['Test Loss'])
        axes[0, 1].set_title('Test Loss Comparison')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Macro F1
        axes[1, 0].bar(df_comparison['Model'], df_comparison['Macro F1'])
        axes[1, 0].set_title('Macro F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Weighted F1
        axes[1, 1].bar(df_comparison['Model'], df_comparison['Weighted F1'])
        axes[1, 1].set_title('Weighted F1 Score Comparison')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.output_config['results_dir'], 
            f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Comparison plots saved to: {plot_path}")

def main():
    """Main function to evaluate models"""
    print("🎯 Model Evaluation Started!")
    
    evaluator = ModelEvaluator()
    
    # Find all model files in the models directory
    models_dir = evaluator.output_config['models_dir']
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        print("Please train some models first!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    
    if not model_files:
        print(f"❌ No model files found in {models_dir}")
        print("Please train some models first!")
        return
    
    print(f"📁 Found {len(model_files)} model files:")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. {model_file}")
    
    # Evaluate all models
    all_results = []
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = model_file.replace('.h5', '')
        
        print(f"\n" + "="*60)
        print(f"Evaluating: {model_name}")
        print("="*60)
        
        try:
            results = evaluator.evaluate_single_model(model_path, model_name)
            
            if results:
                # Create detailed evaluation report
                evaluator.create_evaluation_report(results)
                all_results.append(results)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # Compare all models
    if len(all_results) > 1:
        print(f"\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = evaluator.compare_models(all_results)
        
        # Find best model
        best_model_idx = comparison_df['Test Accuracy'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]
        
        print(f"\n🏆 Best Model: {best_model['Model']}")
        print(f"   Test Accuracy: {best_model['Test Accuracy']:.4f}")
        print(f"   Macro F1: {best_model['Macro F1']:.4f}")
    
    print(f"\n✅ Model evaluation completed!")
    print(f"📁 Results saved to: {os.path.abspath(evaluator.output_config['results_dir'])}")

if __name__ == "__main__":
    main()
