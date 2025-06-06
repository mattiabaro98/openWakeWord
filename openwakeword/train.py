import torch
from torch import optim, nn
import torchinfo
import torchmetrics
import copy
import os
import sys
import tempfile
import uuid
import numpy as np
import scipy
import collections
import argparse
import logging
from tqdm import tqdm
import yaml
from pathlib import Path
import openwakeword
from openwakeword.data import generate_adversarial_texts, augment_clips, mmap_batch_generator
from openwakeword.utils import compute_features_from_generator
from openwakeword.utils import AudioFeatures


class MultiClassWakeWordModel(nn.Module):
    def __init__(self, n_classes=3, input_shape=(16, 96), model_type="dnn",
                 layer_dim=128, n_blocks=1, seconds_per_example=None, 
                 class_names=None, negative_class_label=0):
        """
        Multi-class wake word detection model
        
        Args:
            n_classes (int): Number of classes including negative class
            input_shape (tuple): Input feature shape (time_steps, features)
            model_type (str): "dnn" or "rnn"
            layer_dim (int): Hidden layer dimension
            n_blocks (int): Number of blocks for DNN
            seconds_per_example (float): Duration of each example
            class_names (list): Names of classes (e.g., ['background', 'alexa', 'hey_google'])
            negative_class_label (int): Index of the negative/background class
        """
        super().__init__()

        # Store inputs as attributes
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.seconds_per_example = seconds_per_example
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.negative_class_label = negative_class_label
        
        # Set up class names
        if class_names is None:
            self.class_names = [f'class_{i}' for i in range(n_classes)]
        else:
            assert len(class_names) == n_classes, f"Number of class names ({len(class_names)}) must match n_classes ({n_classes})"
            self.class_names = class_names
        
        # Model tracking
        self.best_models = []
        self.best_model_scores = []
        self.best_val_fp_per_hour = 1000
        self.best_val_accuracy = 0
        self.best_val_recall = 0
        self.best_train_recall = 0

        # Define model architecture
        if model_type == "dnn":
            class FCNBlock(nn.Module):
                def __init__(self, layer_dim, dropout=0.1):
                    super().__init__()
                    self.fcn_layer = nn.Linear(layer_dim, layer_dim)
                    self.relu = nn.ReLU()
                    self.layer_norm = nn.LayerNorm(layer_dim)
                    self.dropout = nn.Dropout(dropout)

                def forward(self, x):
                    return self.dropout(self.relu(self.layer_norm(self.fcn_layer(x))))

            class Net(nn.Module):
                def __init__(self, input_shape, layer_dim, n_blocks=1, n_classes=3, dropout=0.1):
                    super().__init__()
                    self.flatten = nn.Flatten()
                    self.layer1 = nn.Linear(input_shape[0]*input_shape[1], layer_dim)
                    self.relu1 = nn.ReLU()
                    self.layernorm1 = nn.LayerNorm(layer_dim)
                    self.dropout1 = nn.Dropout(dropout)
                    self.blocks = nn.ModuleList([FCNBlock(layer_dim, dropout) for _ in range(n_blocks)])
                    self.last_layer = nn.Linear(layer_dim, n_classes)

                def forward(self, x):
                    x = self.dropout1(self.relu1(self.layernorm1(self.layer1(self.flatten(x)))))
                    for block in self.blocks:
                        x = block(x)
                    x = self.last_layer(x)
                    return x
                    
            self.model = Net(input_shape, layer_dim, n_blocks=n_blocks, n_classes=n_classes)
            
        elif model_type == "rnn":
            class Net(nn.Module):
                def __init__(self, input_shape, n_classes=3, hidden_dim=64, num_layers=2, dropout=0.1):
                    super().__init__()
                    self.lstm = nn.LSTM(input_shape[-1], hidden_dim, num_layers=num_layers, 
                                       bidirectional=True, batch_first=True, dropout=dropout)
                    self.dropout = nn.Dropout(dropout)
                    self.classifier = nn.Linear(hidden_dim * 2, n_classes)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    # Use the last output for classification
                    last_output = lstm_out[:, -1, :]
                    return self.classifier(self.dropout(last_output))
                    
            self.model = Net(input_shape, n_classes)

        # Define metrics for multi-class classification
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)
        self.precision = torchmetrics.Precision(task='multiclass', num_classes=n_classes, average='macro')
        self.recall = torchmetrics.Recall(task='multiclass', num_classes=n_classes, average='macro')
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=n_classes, average='macro')
        
        # Per-class metrics
        self.per_class_precision = torchmetrics.Precision(task='multiclass', num_classes=n_classes, average=None)
        self.per_class_recall = torchmetrics.Recall(task='multiclass', num_classes=n_classes, average=None)
        self.per_class_f1 = torchmetrics.F1Score(task='multiclass', num_classes=n_classes, average=None)

        # Define logging dict (in-memory)
        self.history = collections.defaultdict(list)

        # Define optimizer and loss
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def compute_false_positives(self, predictions, targets, threshold=0.5):
        """
        Compute false positives for multi-class case
        False positive: predicting any positive class when target is negative class
        """
        probs = torch.nn.functional.softmax(predictions, dim=1)
        negative_mask = targets == self.negative_class_label
        
        # Get predictions for negative samples
        neg_probs = probs[negative_mask]
        neg_preds = neg_probs.argmax(dim=1)
        
        # Count false positives (predicting positive class for negative samples)
        fp = (neg_preds != self.negative_class_label).sum()
        return fp

    def compute_positive_class_recall(self, predictions, targets, threshold=0.5):
        """
        Compute recall for positive classes (excluding negative class)
        """
        probs = torch.nn.functional.softmax(predictions, dim=1)
        positive_mask = targets != self.negative_class_label
        
        if positive_mask.sum() == 0:
            return torch.tensor(0.0)
        
        pos_targets = targets[positive_mask]
        pos_probs = probs[positive_mask]
        pos_preds = pos_probs.argmax(dim=1)
        
        # Recall: correct positive predictions / total positives
        correct_positives = (pos_preds == pos_targets).sum()
        total_positives = positive_mask.sum()
        
        return correct_positives.float() / total_positives.float()

    def compute_positive_class_precision(self, predictions, targets, threshold=0.5):
        """
        Compute precision for positive classes
        """
        probs = torch.nn.functional.softmax(predictions, dim=1)
        preds = probs.argmax(dim=1)
        
        # Predictions that are not negative class
        positive_preds_mask = preds != self.negative_class_label
        
        if positive_preds_mask.sum() == 0:
            return torch.tensor(0.0)
        
        # Of the positive predictions, how many are correct?
        pos_preds = preds[positive_preds_mask]
        pos_targets = targets[positive_preds_mask]
        
        correct_pos_preds = (pos_preds == pos_targets).sum()
        total_pos_preds = positive_preds_mask.sum()
        
        return correct_pos_preds.float() / total_pos_preds.float()

    def lr_warmup_cosine_decay(self, global_step, warmup_steps=0, hold=0, total_steps=0,
                               start_lr=0.0, target_lr=1e-3):
        """Learning rate scheduling with warmup and cosine decay"""
        # Cosine decay
        learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold)
                                           / float(total_steps - warmup_steps - hold)))

        # Target LR * progress of warmup
        warmup_lr = target_lr * (global_step / warmup_steps)

        # Choose learning rate based on current step
        if hold > 0:
            learning_rate = np.where(global_step > warmup_steps + hold,
                                     learning_rate, target_lr)

        learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def forward(self, x):
        return self.model(x)

    def summary(self):
        return torchinfo.summary(self.model, input_size=(1,) + self.input_shape, device='cpu')

    def save_model(self, output_path):
        """Save the trained PyTorch model"""
        torch.save(self.model.state_dict(), output_path)
        
    def load_model(self, model_path):
        """Load a trained PyTorch model"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def export_to_onnx(self, output_path, input_names=None, output_names=None):
        """Export model to ONNX format"""
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = self.class_names
            
        # Create a wrapper model that includes softmax for probability output
        class ONNXModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                logits = self.model(x)
                return torch.nn.functional.softmax(logits, dim=1)
        
        onnx_model = ONNXModel(self.model.to("cpu"))
        torch.onnx.export(
            onnx_model,
            torch.rand(1, *self.input_shape),
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=13
        )

    def predict_on_features(self, features, return_probabilities=True):
        """
        Predict on feature tensors
        
        Args:
            features: Input features tensor
            return_probabilities: If True, return softmax probabilities; if False, return logits
        
        Returns:
            Predictions (probabilities or logits) and predicted class indices
        """
        self.model.eval()
        with torch.no_grad():
            if len(features.shape) == 2:
                features = features.unsqueeze(0)
            
            features = features.to(self.device)
            predictions = []
            
            for x in tqdm(features, desc="Predicting on clips"):
                x = x.unsqueeze(0)
                batch = []
                # Sliding window prediction
                for i in range(0, x.shape[1] - 16, 1):
                    batch.append(x[:, i:i+16, :])
                
                if batch:
                    batch = torch.stack(batch, dim=0).squeeze(1)
                    logits = self.model(batch)
                    
                    if return_probabilities:
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        # Take the maximum probability across all windows
                        max_prob, _ = torch.max(probs, dim=0)
                        predictions.append(max_prob.cpu().numpy())
                    else:
                        # Take the maximum logit across all windows
                        max_logit, _ = torch.max(logits, dim=0)
                        predictions.append(max_logit.cpu().numpy())
            
            predictions = np.array(predictions)
            predicted_classes = np.argmax(predictions, axis=1)
            
            return predictions, predicted_classes

    def predict_on_clips(self, clips, return_probabilities=True):
        """
        Predict on raw audio clips
        
        Args:
            clips: Raw audio data
            return_probabilities: If True, return softmax probabilities
            
        Returns:
            Predictions and predicted class indices
        """
        # Get features from clips
        F = AudioFeatures(device='cpu', ncpu=4)
        features = F.embed_clips(clips, batch_size=16)
        
        # Predict on features
        predictions, predicted_classes = self.predict_on_features(
            torch.from_numpy(features), 
            return_probabilities=return_probabilities
        )
        
        return predictions, predicted_classes

    def train_model(self, train_loader, val_loader, false_positive_val_loader=None,
                   max_steps=10000, warmup_steps=1000, hold_steps=2000,
                   lr=0.001, val_steps=None, val_set_hrs=1.0,
                   class_weights=None, save_checkpoints=True):
        """
        Train the multi-class wake word model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            false_positive_val_loader: Loader for false positive validation
            max_steps: Maximum training steps
            warmup_steps: Steps for learning rate warmup
            hold_steps: Steps to hold at target learning rate
            lr: Target learning rate
            val_steps: Steps at which to run validation
            val_set_hrs: Hours of validation data for FP calculation
            class_weights: Weights for each class in loss calculation
            save_checkpoints: Whether to save best model checkpoints
        """
        
        if val_steps is None:
            val_steps = list(range(500, max_steps, 500))
        
        # Set up class weights for imbalanced datasets
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        # Move model to device
        self.model.to(self.device)
        self.model.train()
        
        # Training loop
        step = 0
        train_iter = iter(train_loader)
        
        for step in tqdm(range(max_steps), desc="Training"):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            
            # Update learning rate
            current_lr = self.lr_warmup_cosine_decay(
                step, warmup_steps=warmup_steps, hold=hold_steps,
                total_steps=max_steps, target_lr=lr
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Log training metrics
            self.history["loss"].append(loss.item())
            self.history["lr"].append(current_lr)
            
            # Validation
            if step in val_steps and step > 0:
                self.model.eval()
                val_loss = 0
                all_val_preds = []
                all_val_targets = []
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        x_val, y_val = val_batch[0].to(self.device), val_batch[1].to(self.device)
                        val_logits = self.model(x_val)
                        val_loss += self.loss_fn(val_logits, y_val).item()
                        
                        all_val_preds.append(val_logits)
                        all_val_targets.append(y_val)
                
                # Compute validation metrics
                all_val_preds = torch.cat(all_val_preds, dim=0)
                all_val_targets = torch.cat(all_val_targets, dim=0)
                
                val_accuracy = self.accuracy(all_val_preds, all_val_targets)
                val_precision = self.precision(all_val_preds, all_val_targets)
                val_recall = self.recall(all_val_preds, all_val_targets)
                val_f1 = self.f1_score(all_val_preds, all_val_targets)
                
                # Compute wake word specific metrics
                val_fp = self.compute_false_positives(all_val_preds, all_val_targets)
                val_pos_recall = self.compute_positive_class_recall(all_val_preds, all_val_targets)
                val_pos_precision = self.compute_positive_class_precision(all_val_preds, all_val_targets)
                
                # False positives per hour
                val_fp_per_hour = val_fp.float() / val_set_hrs
                
                # Log validation metrics
                self.history["val_loss"].append(val_loss / len(val_loader))
                self.history["val_accuracy"].append(val_accuracy.item())
                self.history["val_precision"].append(val_precision.item())
                self.history["val_recall"].append(val_recall.item())
                self.history["val_f1"].append(val_f1.item())
                self.history["val_fp_per_hour"].append(val_fp_per_hour.item())
                self.history["val_positive_recall"].append(val_pos_recall.item())
                self.history["val_positive_precision"].append(val_pos_precision.item())
                
                # Per-class metrics
                per_class_prec = self.per_class_precision(all_val_preds, all_val_targets)
                per_class_rec = self.per_class_recall(all_val_preds, all_val_targets)
                per_class_f1_scores = self.per_class_f1(all_val_preds, all_val_targets)
                
                # Log per-class metrics
                for i, class_name in enumerate(self.class_names):
                    self.history[f"val_{class_name}_precision"].append(per_class_prec[i].item())
                    self.history[f"val_{class_name}_recall"].append(per_class_rec[i].item())
                    self.history[f"val_{class_name}_f1"].append(per_class_f1_scores[i].item())
                
                # Save best models based on criteria
                if save_checkpoints:
                    current_score = {
                        'step': step,
                        'val_accuracy': val_accuracy.item(),
                        'val_recall': val_recall.item(),
                        'val_f1': val_f1.item(),
                        'val_fp_per_hour': val_fp_per_hour.item(),
                        'val_positive_recall': val_pos_recall.item(),
                        'val_positive_precision': val_pos_precision.item()
                    }
                    
                    # Save if this is a good model (high recall, low FP rate)
                    if (val_pos_recall > 0.7 and val_fp_per_hour < 1.0) or len(self.best_models) < 5:
                        self.best_models.append(copy.deepcopy(self.model.state_dict()))
                        self.best_model_scores.append(current_score)
                        
                        # Keep only top 10 models
                        if len(self.best_models) > 10:
                            # Sort by F1 score and keep best
                            sorted_indices = sorted(range(len(self.best_model_scores)), 
                                                   key=lambda i: self.best_model_scores[i]['val_f1'], 
                                                   reverse=True)
                            self.best_models = [self.best_models[i] for i in sorted_indices[:10]]
                            self.best_model_scores = [self.best_model_scores[i] for i in sorted_indices[:10]]
                
                # Print validation results
                logging.info(f"Step {step}: Val Acc: {val_accuracy:.4f}, "
                           f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, "
                           f"FP/hr: {val_fp_per_hour:.2f}, Pos Recall: {val_pos_recall:.4f}")
                
                self.model.train()
        
        logging.info("Training completed!")
        return self.history

    def get_best_model(self, criteria='val_f1'):
        """
        Get the best model based on specified criteria
        
        Args:
            criteria: Metric to use for selection ('val_f1', 'val_positive_recall', etc.)
        
        Returns:
            Best model state dict and its scores
        """
        if not self.best_models:
            return None, None
        
        # Find best model based on criteria
        best_idx = max(range(len(self.best_model_scores)), 
                      key=lambda i: self.best_model_scores[i].get(criteria, 0))
        
        return self.best_models[best_idx], self.best_model_scores[best_idx]

    def load_best_model(self, criteria='val_f1'):
        """Load the best model into the current model"""
        best_state_dict, best_scores = self.get_best_model(criteria)
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logging.info(f"Loaded best model with {criteria}: {best_scores[criteria]:.4f}")
        else:
            logging.warning("No best models available to load")

    def evaluate_model(self, test_loader, return_per_class=True):
        """
        Evaluate the model on test data
        
        Args:
            test_loader: Test data loader
            return_per_class: Whether to return per-class metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(x)
                all_preds.append(logits)
                all_targets.append(y)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = {
            'accuracy': self.accuracy(all_preds, all_targets).item(),
            'precision': self.precision(all_preds, all_targets).item(),
            'recall': self.recall(all_preds, all_targets).item(),
            'f1_score': self.f1_score(all_preds, all_targets).item(),
            'false_positives': self.compute_false_positives(all_preds, all_targets).item(),
            'positive_recall': self.compute_positive_class_recall(all_preds, all_targets).item(),
            'positive_precision': self.compute_positive_class_precision(all_preds, all_targets).item()
        }
        
        if return_per_class:
            per_class_prec = self.per_class_precision(all_preds, all_targets)
            per_class_rec = self.per_class_recall(all_preds, all_targets)
            per_class_f1_scores = self.per_class_f1(all_preds, all_targets)
            
            for i, class_name in enumerate(self.class_names):
                metrics[f'{class_name}_precision'] = per_class_prec[i].item()
                metrics[f'{class_name}_recall'] = per_class_rec[i].item()
                metrics[f'{class_name}_f1'] = per_class_f1_scores[i].item()
        
        return metrics

# Separate function to convert onnx models to tflite format
def convert_onnx_to_tflite(onnx_model_path, output_path):
    """Converts an ONNX version of an openwakeword model to the Tensorflow tflite format."""
    # imports
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # Convert to tflite from onnx model
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model, device="CPU")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tf_rep.export_graph(os.path.join(tmp_dir, "tf_model"))
        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(tmp_dir, "tf_model"))
        tflite_model = converter.convert()

        logging.info(f"####\nSaving tflite mode to '{output_path}'")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

    return None

if __name__ == '__main__':
    # Get training config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_config",
        help="The path to the training config file (required)",
        type=str,
        required=True
    )
    parser.add_argument(
        "--generate_clips",
        help="Execute the synthetic data generation process",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--augment_clips",
        help="Execute the synthetic data augmentation process",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing openwakeword features when the --augment_clips flag is used",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--train_model",
        help="Execute the model training process",
        action="store_true",
        default="False",
        required=False
    )

    args = parser.parse_args()
    config = yaml.load(open(args.training_config, 'r').read(), yaml.Loader)

    # imports Piper for synthetic sample generation
    sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
    from generate_samples import generate_samples

    # Define output locations
    config["output_dir"] = os.path.abspath(config["output_dir"])
    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])
    if not os.path.exists(os.path.join(config["output_dir"], config["model_name"])):
        os.mkdir(os.path.join(config["output_dir"], config["model_name"]))

    # Create directories for each class
    class_names = config["class_names"]  # e.g., ['background', 'alexa', 'hey_google', 'jarvis']
    n_classes = len(class_names)
    negative_class_label = config.get("negative_class_label", 0)
    
    # Create output directories for each class
    class_train_dirs = {}
    class_test_dirs = {}
    
    for i, class_name in enumerate(class_names):
        if i != negative_class_label:  # Skip negative class (will be generated separately)
            class_train_dirs[class_name] = os.path.join(config["output_dir"], config["model_name"], f"{class_name}_train")
            class_test_dirs[class_name] = os.path.join(config["output_dir"], config["model_name"], f"{class_name}_test")
    
    # Negative class directories
    negative_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_train")
    negative_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_test")
    feature_save_dir = os.path.join(config["output_dir"], config["model_name"])

    # Get paths for impulse response and background audio files
    rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]
    background_paths = []
    if len(config["background_paths_duplication_rate"]) != len(config["background_paths"]):
        config["background_paths_duplication_rate"] = [1]*len(config["background_paths"])
    for background_path, duplication_rate in zip(config["background_paths"], config["background_paths_duplication_rate"]):
        background_paths.extend([i.path for i in os.scandir(background_path)]*duplication_rate)

    if args.generate_clips is True:
        # Generate positive clips for each class
        for i, class_name in enumerate(class_names):
            if i == negative_class_label:  # Skip negative class
                continue
                
            target_phrases = config["target_phrases"][class_name]  # Dictionary of phrases per class
            
            # Generate training clips for this class
            logging.info("#"*50 + f"\nGenerating {class_name} clips for training\n" + "#"*50)
            if not os.path.exists(class_train_dirs[class_name]):
                os.mkdir(class_train_dirs[class_name])
            n_current_samples = len(os.listdir(class_train_dirs[class_name]))
            if n_current_samples <= 0.95*config["n_samples"]:
                generate_samples(
                    text=target_phrases, max_samples=config["n_samples"]-n_current_samples,
                    batch_size=config["tts_batch_size"],
                    noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.75, 1.0, 1.25],
                    output_dir=class_train_dirs[class_name], auto_reduce_batch_size=True,
                    file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
                )
                torch.cuda.empty_cache()
            else:
                logging.warning(f"Skipping generation of {class_name} clips for training, as ~{config['n_samples']} already exist")

            # Generate testing clips for this class
            logging.info("#"*50 + f"\nGenerating {class_name} clips for testing\n" + "#"*50)
            if not os.path.exists(class_test_dirs[class_name]):
                os.mkdir(class_test_dirs[class_name])
            n_current_samples = len(os.listdir(class_test_dirs[class_name]))
            if n_current_samples <= 0.95*config["n_samples_val"]:
                generate_samples(text=target_phrases, max_samples=config["n_samples_val"]-n_current_samples,
                                 batch_size=config["tts_batch_size"],
                                 noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.75, 1.0, 1.25],
                                 output_dir=class_test_dirs[class_name], auto_reduce_batch_size=True)
                torch.cuda.empty_cache()
            else:
                logging.warning(f"Skipping generation of {class_name} clips for testing, as ~{config['n_samples_val']} already exist")

        # Generate adversarial negative clips for training
        logging.info("#"*50 + "\nGenerating negative clips for training\n" + "#"*50)
        if not os.path.exists(negative_train_output_dir):
            os.mkdir(negative_train_output_dir)
        n_current_samples = len(os.listdir(negative_train_output_dir))
        if n_current_samples <= 0.95*config["n_samples"]:
            adversarial_texts = config["custom_negative_phrases"]
            
            # Generate adversarial texts for all positive classes
            all_target_phrases = []
            for class_name in class_names:
                if class_names.index(class_name) != negative_class_label:
                    all_target_phrases.extend(config["target_phrases"][class_name])
            
            for target_phrase in all_target_phrases:
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=target_phrase,
                    N=config["n_samples"]//len(all_target_phrases),
                    include_partial_phrase=1.0,
                    include_input_words=0.2))
            
            generate_samples(text=adversarial_texts, max_samples=config["n_samples"]-n_current_samples,
                             batch_size=config["tts_batch_size"]//7,
                             noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.75, 1.0, 1.25],
                             output_dir=negative_train_output_dir, auto_reduce_batch_size=True,
                             file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
                             )
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of negative clips for training, as ~{config['n_samples']} already exist")

        # Generate adversarial negative clips for testing
        logging.info("#"*50 + "\nGenerating negative clips for testing\n" + "#"*50)
        if not os.path.exists(negative_test_output_dir):
            os.mkdir(negative_test_output_dir)
        n_current_samples = len(os.listdir(negative_test_output_dir))
        if n_current_samples <= 0.95*config["n_samples_val"]:
            adversarial_texts = config["custom_negative_phrases"]
            
            # Generate adversarial texts for all positive classes
            all_target_phrases = []
            for class_name in class_names:
                if class_names.index(class_name) != negative_class_label:
                    all_target_phrases.extend(config["target_phrases"][class_name])
            
            for target_phrase in all_target_phrases:
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=target_phrase,
                    N=config["n_samples_val"]//len(all_target_phrases),
                    include_partial_phrase=1.0,
                    include_input_words=0.2))
            
            generate_samples(text=adversarial_texts, max_samples=config["n_samples_val"]-n_current_samples,
                             batch_size=config["tts_batch_size"]//7,
                             noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.75, 1.0, 1.25],
                             output_dir=negative_test_output_dir, auto_reduce_batch_size=True)
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of negative clips for testing, as ~{config['n_samples_val']} already exist")

    # Set the total length of the training clips based on the ~median generated clip duration
    n = 50  # sample size
    # Use clips from the first positive class to determine duration
    first_positive_class = [name for i, name in enumerate(class_names) if i != negative_class_label][0]
    positive_clips = [str(i) for i in Path(class_test_dirs[first_positive_class]).glob("*.wav")]
    duration_in_samples = []
    for i in range(min(n, len(positive_clips))):
        sr, dat = scipy.io.wavfile.read(positive_clips[np.random.randint(0, len(positive_clips))])
        duration_in_samples.append(len(dat))

    config["total_length"] = int(round(np.median(duration_in_samples)/1000)*1000) + 12000  # add 750 ms to clip duration as buffer
    if config["total_length"] < 32000:
        config["total_length"] = 32000  # set a minimum of 32000 samples (2 seconds)
    elif abs(config["total_length"] - 32000) <= 4000:
        config["total_length"] = 32000

    # Do Data Augmentation
    if args.augment_clips is True:
        feature_files = {}
        for class_name in class_names:
            if class_names.index(class_name) != negative_class_label:
                feature_files[f"{class_name}_train"] = os.path.join(feature_save_dir, f"{class_name}_features_train.npy")
                feature_files[f"{class_name}_test"] = os.path.join(feature_save_dir, f"{class_name}_features_test.npy")
        
        feature_files["negative_train"] = os.path.join(feature_save_dir, "negative_features_train.npy")
        feature_files["negative_test"] = os.path.join(feature_save_dir, "negative_features_test.npy")
        
        # Check if any feature files are missing or if overwrite is requested
        need_to_generate = args.overwrite or any(not os.path.exists(path) for path in feature_files.values())
        
        if need_to_generate:
            # Generate augmented clips and features for each class
            for class_name in class_names:
                if class_names.index(class_name) != negative_class_label:
                    # Training data
                    clips_train = [str(i) for i in Path(class_train_dirs[class_name]).glob("*.wav")] * config["augmentation_rounds"]
                    clips_train_generator = augment_clips(clips_train, total_length=config["total_length"],
                                                         batch_size=config["augmentation_batch_size"],
                                                         background_clip_paths=background_paths,
                                                         RIR_paths=rir_paths)

                    # Testing data
                    clips_test = [str(i) for i in Path(class_test_dirs[class_name]).glob("*.wav")] * config["augmentation_rounds"]
                    clips_test_generator = augment_clips(clips_test, total_length=config["total_length"],
                                                        batch_size=config["augmentation_batch_size"],
                                                        background_clip_paths=background_paths,
                                                        RIR_paths=rir_paths)

                    # Compute features for this class
                    logging.info("#"*50 + f"\nComputing openwakeword features for {class_name} samples\n" + "#"*50)
                    n_cpus = os.cpu_count()
                    if n_cpus is None:
                        n_cpus = 1
                    else:
                        n_cpus = n_cpus//2

                    compute_features_from_generator(clips_train_generator, n_total=len(os.listdir(class_train_dirs[class_name])),
                                                    clip_duration=config["total_length"],
                                                    output_file=feature_files[f"{class_name}_train"],
                                                    device="gpu" if torch.cuda.is_available() else "cpu",
                                                    ncpu=n_cpus if not torch.cuda.is_available() else 1)

                    compute_features_from_generator(clips_test_generator, n_total=len(os.listdir(class_test_dirs[class_name])),
                                                    clip_duration=config["total_length"],
                                                    output_file=feature_files[f"{class_name}_test"],
                                                    device="gpu" if torch.cuda.is_available() else "cpu",
                                                    ncpu=n_cpus if not torch.cuda.is_available() else 1)

            # Generate negative class features
            negative_clips_train = [str(i) for i in Path(negative_train_output_dir).glob("*.wav")] * config["augmentation_rounds"]
            negative_clips_train_generator = augment_clips(negative_clips_train, total_length=config["total_length"],
                                                           batch_size=config["augmentation_batch_size"],
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths)

            negative_clips_test = [str(i) for i in Path(negative_test_output_dir).glob("*.wav")] * config["augmentation_rounds"]
            negative_clips_test_generator = augment_clips(negative_clips_test, total_length=config["total_length"],
                                                          batch_size=config["augmentation_batch_size"],
                                                          background_clip_paths=background_paths,
                                                          RIR_paths=rir_paths)

            logging.info("#"*50 + "\nComputing openwakeword features for negative samples\n" + "#"*50)
            compute_features_from_generator(negative_clips_train_generator, n_total=len(os.listdir(negative_train_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=feature_files["negative_train"],
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            compute_features_from_generator(negative_clips_test_generator, n_total=len(os.listdir(negative_test_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=feature_files["negative_test"],
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)
        else:
            logging.warning("Openwakeword features already exist, skipping data augmentation and feature generation")

    # Create multi-class openwakeword model
    if args.train_model is True:
        F = openwakeword.utils.AudioFeatures(device='cpu')
        
        # Get input shape from one of the feature files
        first_feature_file = os.path.join(feature_save_dir, f"{[name for i, name in enumerate(class_names) if i != negative_class_label][0]}_features_test.npy")
        input_shape = np.load(first_feature_file).shape[1:]

        oww = MultiClassWakeWordModel(
            n_classes=n_classes, 
            input_shape=input_shape, 
            model_type=config["model_type"],
            layer_dim=config["layer_size"], 
            seconds_per_example=1280*input_shape[0]/16000,
            class_names=class_names,
            negative_class_label=negative_class_label
        )

        # Create data transform function for batch generation
        def f(x, n=input_shape[0]):
            """Simple transformation function to ensure data is the appropriate shape for the model size"""
            if n > x.shape[1] or n < x.shape[1]:
                x = np.vstack(x)
                new_batch = np.array([x[i:i+n, :] for i in range(0, x.shape[0]-n, n)])
            else:
                return x
            return new_batch

        # Create feature data files dictionary for training
        training_feature_files = {}
        for i, class_name in enumerate(class_names):
            if i != negative_class_label:
                training_feature_files[class_name] = os.path.join(feature_save_dir, f"{class_name}_features_train.npy")
            else:
                training_feature_files[class_name] = os.path.join(feature_save_dir, "negative_features_train.npy")

        # Add any additional feature data files from config
        if "feature_data_files" in config:
            training_feature_files.update(config["feature_data_files"])

        # Create label transforms for multi-class classification
        data_transforms = {key: f for key in training_feature_files.keys()}
        label_transforms = {}
        for i, class_name in enumerate(class_names):
            label_transforms[class_name] = lambda x, class_idx=i: [class_idx for _ in x]

        # Make PyTorch data loaders for training and validation data
        batch_generator = mmap_batch_generator(
            training_feature_files,
            n_per_class=config["batch_n_per_class"],
            data_transform_funcs=data_transforms,
            label_transform_funcs=label_transforms
        )

        class IterDataset(torch.utils.data.IterableDataset):
            def __init__(self, generator):
                self.generator = generator

            def __iter__(self):
                return self.generator

        n_cpus = os.cpu_count()
        if n_cpus is None:
            n_cpus = 1
        else:
            n_cpus = n_cpus//2
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)

        # Prepare validation data
        X_val_fp = np.load(config["false_positive_validation_data_path"])
        X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])
        X_val_fp_labels = np.full(X_val_fp.shape[0], negative_class_label).astype(np.float32)  # All false positives are negative class
        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
            batch_size=len(X_val_fp_labels)
        )

        # Combine all test features and labels
        all_val_features = []
        all_val_labels = []
        
        for i, class_name in enumerate(class_names):
            if i != negative_class_label:
                class_features = np.load(os.path.join(feature_save_dir, f"{class_name}_features_test.npy"))
            else:
                class_features = np.load(os.path.join(feature_save_dir, "negative_features_test.npy"))
            
            all_val_features.append(class_features)
            all_val_labels.extend([i] * class_features.shape[0])

        X_val_combined = np.vstack(all_val_features)
        labels_combined = np.array(all_val_labels).astype(np.float32)

        X_val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_val_combined),
                torch.from_numpy(labels_combined)
            ),
            batch_size=len(labels_combined)
        )

        # Run auto training (you'll need to implement this for multi-class)
        best_model = oww.auto_train(
            X_train=X_train,
            X_val=X_val,
            false_positive_val_data=X_val_fp,
            steps=config["steps"],
            max_negative_weight=config.get("max_negative_weight", 1.0),
            target_fp_per_hour=config.get("target_false_positives_per_hour", 1.0),
        )

        # Export the trained model to onnx
        oww.export_model(model=best_model, model_name=config["model_name"], output_dir=config["output_dir"])

        # Convert the model from onnx to tflite format
        convert_onnx_to_tflite(os.path.join(config["output_dir"], config["model_name"] + ".onnx"),
                               os.path.join(config["output_dir"], config["model_name"] + ".tflite"))
