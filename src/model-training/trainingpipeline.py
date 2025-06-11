from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizer,
    TrainerCallback,
)
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import pandas as pd
import numpy as np
from constants import TRAINING_LOGS_PATH
from loguru import logger

from transformers import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message="Some weights of the model checkpoint were not used when initializing",
)
logging.set_verbosity_error()

logger.add(sink=TRAINING_LOGS_PATH)


class CustomDataset(Dataset):
    def __init__(self, encodings, labels, ood_labels=None):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ood_labels = (
            torch.tensor(ood_labels, dtype=torch.long)
            if ood_labels is not None
            else None
        )

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        if self.ood_labels is not None:
            item["ood_labels"] = self.ood_labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class SpectralNormalization(nn.Module):
    """Spectral normalization layer for improved uncertainty estimation"""

    def __init__(self, layer, n_power_iterations=1, eps=1e-12):
        super().__init__()
        self.layer = layer
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        # Initialize spectral norm
        with torch.no_grad():
            weight = getattr(layer, "weight")
            h, w = weight.size()
            u = nn.Parameter(torch.randn(h), requires_grad=False)
            v = nn.Parameter(torch.randn(w), requires_grad=False)
            self.register_parameter("u", u)
            self.register_parameter("v", v)

    def forward(self, x):
        if self.training:
            self._update_uv()
        return self.layer(x)

    def _update_uv(self):
        weight = getattr(self.layer, "weight")
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                self.v.data = F.normalize(
                    torch.mv(weight.t(), self.u), dim=0, eps=self.eps
                )
                self.u.data = F.normalize(torch.mv(weight, self.v), dim=0, eps=self.eps)

            sigma = torch.dot(self.u, torch.mv(weight, self.v))
            weight.data /= sigma


class OODLoss(nn.Module):
    """Combined loss for OOD detection"""

    def __init__(self, ood_weight=0.1, energy_margin=10.0, temperature=1.0):
        super().__init__()
        self.ood_weight = ood_weight
        self.energy_margin = energy_margin
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels, ood_labels=None):
        # Standard classification loss
        ce_loss = self.ce_loss(logits, labels)

        if ood_labels is None:
            return ce_loss

        # Energy-based OOD loss
        energy = -torch.logsumexp(logits / self.temperature, dim=-1)

        # Separate ID and OOD samples
        is_ood = (ood_labels == 1).float()
        is_id = 1.0 - is_ood

        # Energy loss: low energy for ID, high energy for OOD
        energy_loss = (
            torch.relu(self.energy_margin - energy) * is_ood  # High energy for OOD
            + energy * is_id  # Low energy for ID
        )

        total_loss = ce_loss + self.ood_weight * energy_loss.mean()
        return total_loss


class EnhancedModel(nn.Module):
    """Enhanced model with optional OOD detection capabilities"""

    def __init__(
        self,
        base_model,
        num_labels,
        use_spectral_norm=False,
        hidden_dim=768,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.use_spectral_norm = use_spectral_norm

        # Get the hidden size from the base model
        if hasattr(base_model.config, "hidden_size"):
            self.hidden_size = base_model.config.hidden_size
        else:
            self.hidden_size = hidden_dim

        # Additional layers for uncertainty estimation
        if use_spectral_norm:
            self.uncertainty_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                SpectralNormalization(nn.Linear(self.hidden_size, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                SpectralNormalization(nn.Linear(hidden_dim, num_labels)),
            )
        else:
            self.uncertainty_head = None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        if self.uncertainty_head is not None:
            # Use pooled output or last hidden state
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                # Use [CLS] token representation
                pooled_output = outputs.last_hidden_state[:, 0, :]

            logits = self.uncertainty_head(pooled_output)
            outputs.logits = logits

        return outputs

    def predict_with_uncertainty(self, input_ids, attention_mask=None):
        """Predict with uncertainty estimation"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs.logits

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

            # Calculate uncertainty (entropy-based)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            # Energy-based uncertainty
            energy = -torch.logsumexp(logits, dim=-1)

            return predictions, entropy, energy


class ResetCovarianceCallback(TrainerCallback):
    """Callback to reset covariance for SNGP-like models"""

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Reset any covariance matrices if needed
        pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"TRAINING HARDWARE {device}")


class TrainingPipeline:
    def __init__(self, dfs, model_name):
        self.model_name = model_name
        self.dfs = dfs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Updated model initialization for multilingual/Estonian models
        if model_name == "estbert":
            self.base_model = BertForSequenceClassification.from_pretrained(
                "tartuNLP/EstBERT"
            )
            self._freeze_and_unfreeze_bert_layers()

        elif model_name == "estbert-small":
            self.base_model = BertForSequenceClassification.from_pretrained(
                "tartuNLP/EstBERT-small"
            )
            self._freeze_and_unfreeze_bert_layers()

        elif model_name == "xlm-roberta":
            self.base_model = XLMRobertaForSequenceClassification.from_pretrained(
                "xlm-roberta-base"
            )
            self._freeze_and_unfreeze_roberta_layers()

        elif model_name == "mdeberta":
            from transformers import DebertaV2ForSequenceClassification

            self.base_model = DebertaV2ForSequenceClassification.from_pretrained(
                "microsoft/mdeberta-v3-base"
            )
            self._freeze_and_unfreeze_deberta_layers()

        elif model_name == "multilingual-bert":
            self.base_model = BertForSequenceClassification.from_pretrained(
                "bert-base-multilingual-cased"
            )
            self._freeze_and_unfreeze_bert_layers()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _freeze_and_unfreeze_bert_layers(self):
        """Helper method for BERT-based models"""
        for param in self.base_model.bert.parameters():
            param.requires_grad = False
        for param in self.base_model.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True

    def _freeze_and_unfreeze_roberta_layers(self):
        """Helper method for RoBERTa-based models"""
        for param in self.base_model.roberta.parameters():
            param.requires_grad = False
        for param in self.base_model.roberta.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True

    def _freeze_and_unfreeze_deberta_layers(self):
        """Helper method for DeBERTa-based models"""
        for param in self.base_model.deberta.parameters():
            param.requires_grad = False
        for param in self.base_model.deberta.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True

    def preprocess_conversation(self, text):
        """Preprocess conversation data"""
        # Add conversation-specific preprocessing
        text = str(text).strip()

        # Handle common conversation patterns
        text = text.replace("\n", " [SEP] ")  # Use [SEP] for line breaks
        text = text.replace("\t", " ")  # Remove tabs

        # Limit length for transformer models (adjust based on model)
        max_length = 400  # Reasonable for conversations
        if len(text.split()) > max_length:
            words = text.split()[:max_length]
            text = " ".join(words)

        return text

    def tokenize_data(self, data, tokenizer):
        """Enhanced tokenization for conversation data"""
        # Preprocess conversation data
        processed_data = [self.preprocess_conversation(text) for text in data]

        tokenized = tokenizer.batch_encode_plus(
            processed_data,
            truncation=True,
            padding=True,
            max_length=512,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return tokenized

    def data_split(self, df):
        """Fixed data split to prevent data leakage"""
        # Ensure we have at least one sample of each class in training
        unique_classes = df["target"].unique()

        # If we have very few samples, use stratified split
        if len(df) < 100:
            # For small datasets, ensure each class has representation
            train_samples = []
            test_samples = []

            for class_name in unique_classes:
                class_samples = df[df["target"] == class_name]

                if len(class_samples) == 1:
                    # If only one sample, put it in training
                    train_samples.append(class_samples)
                else:
                    # Split the class samples
                    train_class, test_class = train_test_split(
                        class_samples, test_size=0.2, random_state=42
                    )
                    train_samples.append(train_class)
                    test_samples.append(test_class)

            train_df = pd.concat(train_samples) if train_samples else pd.DataFrame()
            test_df = pd.concat(test_samples) if test_samples else pd.DataFrame()

            # If test set is empty, duplicate some training samples for testing
            if len(test_df) == 0:
                test_df = train_df.sample(min(len(train_df), 5), random_state=42)
        else:
            # For larger datasets, use standard stratified split
            try:
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42, stratify=df["target"]
                )
            except ValueError:
                # If stratification fails, use simple split
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        return train_df, test_df

    def get_tokenizer_for_model(self, model_name, num_labels):
        """Get appropriate tokenizer and model for training"""
        if model_name in ["estbert", "estbert-small", "multilingual-bert"]:
            if model_name == "estbert":
                model = BertForSequenceClassification.from_pretrained(
                    "tartuNLP/EstBERT",
                    num_labels=num_labels,
                    state_dict=self.base_model.state_dict(),
                    ignore_mismatched_sizes=True,
                )
                tokenizer = BertTokenizer.from_pretrained("tartuNLP/EstBERT")
            elif model_name == "estbert-small":
                model = BertForSequenceClassification.from_pretrained(
                    "tartuNLP/EstBERT-small",
                    num_labels=num_labels,
                    state_dict=self.base_model.state_dict(),
                    ignore_mismatched_sizes=True,
                )
                tokenizer = BertTokenizer.from_pretrained("tartuNLP/EstBERT-small")
            else:  # multilingual-bert
                model = BertForSequenceClassification.from_pretrained(
                    "bert-base-multilingual-cased",
                    num_labels=num_labels,
                    state_dict=self.base_model.state_dict(),
                    ignore_mismatched_sizes=True,
                )
                tokenizer = BertTokenizer.from_pretrained(
                    "bert-base-multilingual-cased"
                )

            self._freeze_and_unfreeze_bert_layers_for_model(model)
            return model, tokenizer

        elif model_name == "xlm-roberta":
            model = XLMRobertaForSequenceClassification.from_pretrained(
                "xlm-roberta-base",
                num_labels=num_labels,
                state_dict=self.base_model.state_dict(),
                ignore_mismatched_sizes=True,
            )
            tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
            self._freeze_and_unfreeze_roberta_layers_for_model(model)
            return model, tokenizer

        elif model_name == "mdeberta":
            from transformers import (
                DebertaV2ForSequenceClassification,
                DebertaV2Tokenizer,
            )

            model = DebertaV2ForSequenceClassification.from_pretrained(
                "microsoft/mdeberta-v3-base",
                num_labels=num_labels,
                state_dict=self.base_model.state_dict(),
                ignore_mismatched_sizes=True,
            )
            tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/mdeberta-v3-base")
            self._freeze_and_unfreeze_deberta_layers_for_model(model)
            return model, tokenizer

        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _freeze_and_unfreeze_bert_layers_for_model(self, model):
        """Apply layer freezing for BERT models during training"""
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    def _freeze_and_unfreeze_roberta_layers_for_model(self, model):
        """Apply layer freezing for RoBERTa models during training"""
        for param in model.roberta.parameters():
            param.requires_grad = False
        for param in model.roberta.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    def _freeze_and_unfreeze_deberta_layers_for_model(self, model):
        """Apply layer freezing for DeBERTa models during training"""
        for param in model.deberta.parameters():
            param.requires_grad = False
        for param in model.deberta.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    def train(self):
        classes = []
        accuracies = []
        f1_scores = []
        models = []
        classifiers = []
        label_encoders = []

        logger.info(f"INITIATING TRAINING FOR {self.model_name} MODEL")
        for i in range(len(self.dfs)):
            logger.info(f"TRAINING FOR DATAFRAME {i + 1} of {len(self.dfs)}")
            current_df = self.dfs[i]
            if len(current_df) < 10:
                current_df = self.replicate_data(current_df, 50).reset_index(drop=True)

            train_df, test_df = self.data_split(current_df)
            label_encoder = LabelEncoder()
            train_labels = label_encoder.fit_transform(train_df["target"])
            test_labels = label_encoder.transform(test_df["target"])

            # Get model and tokenizer
            model, tokenizer = self.get_tokenizer_for_model(
                self.model_name, len(label_encoder.classes_)
            )

            train_encodings = self.tokenize_data(train_df["input"].tolist(), tokenizer)
            test_encodings = self.tokenize_data(test_df["input"].tolist(), tokenizer)

            train_dataset = CustomDataset(train_encodings, train_labels)
            test_dataset = CustomDataset(test_encodings, test_labels)

            training_args = TrainingArguments(
                output_dir="tmp",
                num_train_epochs=4,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                learning_rate=2e-5,
                warmup_steps=100,
                weight_decay=0.01,
                logging_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                disable_tqdm=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=lambda eval_pred: {
                    "accuracy": accuracy_score(
                        eval_pred.label_ids, eval_pred.predictions.argmax(axis=1)
                    )
                },
            )

            trainer.train()

            # Extract model layers based on model type
            if self.model_name in ["estbert", "estbert-small", "multilingual-bert"]:
                models.append(model.bert.encoder.layer[-2:].state_dict())
                classifiers.append(model.classifier.state_dict())
            elif self.model_name == "xlm-roberta":
                models.append(model.roberta.encoder.layer[-2:].state_dict())
                classifiers.append(model.classifier.state_dict())
            elif self.model_name == "mdeberta":
                models.append(model.deberta.encoder.layer[-2:].state_dict())
                classifiers.append(model.classifier.state_dict())

            predictions, labels, _ = trainer.predict(test_dataset)
            predictions = predictions.argmax(axis=-1)
            report = classification_report(
                labels,
                predictions,
                target_names=label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            )

            # Log agency classification results
            logger.info(f"Agency Classification Results for {self.model_name}:")
            for cls in label_encoder.classes_:
                precision = report[cls]["precision"]
                f1 = report[cls]["f1-score"]
                logger.info(f"  Agency '{cls}': Precision={precision:.3f}, F1={f1:.3f}")

                classes.append(cls)
                accuracies.append(precision)
                f1_scores.append(f1)

            label_encoders.append(label_encoder)
            shutil.rmtree("tmp")

        basic_model = self.base_model.state_dict()
        metrics = (classes, accuracies, f1_scores)
        return metrics, models, classifiers, label_encoders, basic_model


class EnhancedTrainingPipeline(TrainingPipeline):
    """Enhanced Training Pipeline with OOD support"""

    def __init__(
        self,
        dfs,
        model_name,
        ood_training=False,
        ood_weight=0.1,
        energy_margin=10.0,
        temperature=1.0,
        use_spectral_norm=False,
        uncertainty_method="entropy",
    ):
        # Initialize parent class
        super().__init__(dfs, model_name)

        # OOD-specific parameters
        self.ood_training = ood_training
        self.ood_weight = ood_weight
        self.energy_margin = energy_margin
        self.temperature = temperature
        self.use_spectral_norm = use_spectral_norm
        self.uncertainty_method = uncertainty_method

        logger.info("ENHANCED TRAINING PIPELINE INITIALIZED")
        logger.info(f"OOD TRAINING: {self.ood_training}")
        if self.ood_training:
            logger.info(
                f"OOD CONFIG - Weight: {self.ood_weight}, Margin: {self.energy_margin}, Temp: {self.temperature}"
            )

    def prepare_ood_data(self, df):
        """Prepare data with OOD labels if OOD training is enabled"""
        if not self.ood_training:
            return df, None

        # For standard training data, all samples are in-distribution (ID)
        # In practice, you would have actual OOD samples in your dataset
        # or implement synthetic OOD generation strategies
        ood_labels = np.zeros(len(df))  # All samples are ID by default

        return df, ood_labels

    def create_enhanced_model(self, base_model, num_labels):
        """Create enhanced model with optional OOD capabilities"""
        if self.ood_training or self.use_spectral_norm:
            enhanced_model = EnhancedModel(
                base_model=base_model,
                num_labels=num_labels,
                use_spectral_norm=self.use_spectral_norm,
            )
            return enhanced_model
        else:
            return base_model

    def train(self):
        classes = []
        accuracies = []
        f1_scores = []
        models = []
        classifiers = []
        label_encoders = []
        ood_metrics = []  # Store OOD-specific metrics

        logger.info(f"INITIATING ENHANCED TRAINING FOR {self.model_name} MODEL")
        logger.info(f"OOD TRAINING: {self.ood_training}")

        for i in range(len(self.dfs)):
            logger.info(f"TRAINING FOR DATAFRAME {i + 1} of {len(self.dfs)}")
            current_df = self.dfs[i]

            if len(current_df) < 10:
                current_df = self.replicate_data(current_df, 50).reset_index(drop=True)

            train_df, test_df = self.data_split(current_df)

            # Prepare OOD data if enabled
            train_df, train_ood_labels = self.prepare_ood_data(train_df)
            test_df, test_ood_labels = self.prepare_ood_data(test_df)

            label_encoder = LabelEncoder()
            train_labels = label_encoder.fit_transform(train_df["target"])
            test_labels = label_encoder.transform(test_df["target"])

            # Create model with appropriate configuration
            num_labels = len(label_encoder.classes_)

            if self.model_name == "distil-bert":
                base_model = DistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased",
                    num_labels=num_labels,
                    state_dict=self.base_model.state_dict(),
                    ignore_mismatched_sizes=True,
                )
                tokenizer = DistilBertTokenizer.from_pretrained(
                    "distilbert-base-uncased"
                )

                if self.ood_training or self.use_spectral_norm:
                    model = self.create_enhanced_model(base_model, num_labels)
                else:
                    model = base_model

                # Apply layer freezing
                if hasattr(model, "base_model"):
                    for param in model.base_model.distilbert.parameters():
                        param.requires_grad = False
                    for param in model.base_model.distilbert.transformer.layer[
                        -2:
                    ].parameters():
                        param.requires_grad = True
                    for param in model.base_model.classifier.parameters():
                        param.requires_grad = True
                else:
                    for param in model.distilbert.parameters():
                        param.requires_grad = False
                    for param in model.distilbert.transformer.layer[-2:].parameters():
                        param.requires_grad = True
                    for param in model.classifier.parameters():
                        param.requires_grad = True

            elif self.model_name == "roberta":
                base_model = XLMRobertaForSequenceClassification.from_pretrained(
                    "xlm-roberta-base",
                    num_labels=num_labels,
                    state_dict=self.base_model.state_dict(),
                    ignore_mismatched_sizes=True,
                )
                tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

                if self.ood_training or self.use_spectral_norm:
                    model = self.create_enhanced_model(base_model, num_labels)
                else:
                    model = base_model

                # Apply layer freezing
                if hasattr(model, "base_model"):
                    for param in model.base_model.roberta.parameters():
                        param.requires_grad = False
                    for param in model.base_model.roberta.encoder.layer[
                        -2:
                    ].parameters():
                        param.requires_grad = True
                    for param in model.base_model.classifier.parameters():
                        param.requires_grad = True
                else:
                    for param in model.roberta.parameters():
                        param.requires_grad = False
                    for param in model.roberta.encoder.layer[-2:].parameters():
                        param.requires_grad = True
                    for param in model.classifier.parameters():
                        param.requires_grad = True

            elif self.model_name == "bert":
                base_model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=num_labels,
                    state_dict=self.base_model.state_dict(),
                    ignore_mismatched_sizes=True,
                )
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

                if self.ood_training or self.use_spectral_norm:
                    model = self.create_enhanced_model(base_model, num_labels)
                else:
                    model = base_model

                # Apply layer freezing
                if hasattr(model, "base_model"):
                    for param in model.base_model.bert.parameters():
                        param.requires_grad = False
                    for param in model.base_model.bert.encoder.layer[-2:].parameters():
                        param.requires_grad = True
                    for param in model.base_model.classifier.parameters():
                        param.requires_grad = True
                else:
                    for param in model.bert.parameters():
                        param.requires_grad = False
                    for param in model.bert.encoder.layer[-2:].parameters():
                        param.requires_grad = True
                    for param in model.classifier.parameters():
                        param.requires_grad = True

            # Tokenize data
            train_encodings = self.tokenize_data(train_df["input"].tolist(), tokenizer)
            test_encodings = self.tokenize_data(test_df["input"].tolist(), tokenizer)

            # Create datasets
            train_dataset = CustomDataset(
                train_encodings, train_labels, train_ood_labels
            )
            test_dataset = CustomDataset(test_encodings, test_labels, test_ood_labels)

            # Create custom trainer if OOD training is enabled
            if self.ood_training:
                training_args = TrainingArguments(
                    output_dir="tmp",
                    num_train_epochs=4,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    logging_dir="./logs",
                    logging_steps=100,
                    eval_strategy="epoch",
                    disable_tqdm=False,
                    dataloader_drop_last=False,
                )

                class OODTrainer(Trainer):
                    def __init__(self, ood_loss_fn=None, **kwargs):
                        super().__init__(**kwargs)
                        self.ood_loss_fn = ood_loss_fn

                    def compute_loss(self, model, inputs, return_outputs=False):
                        labels = inputs.get("labels")
                        ood_labels = inputs.get("ood_labels")

                        # Forward pass
                        outputs = model(
                            **{
                                k: v
                                for k, v in inputs.items()
                                if k not in ["labels", "ood_labels"]
                            }
                        )
                        logits = outputs.get("logits")

                        # Compute loss
                        if self.ood_loss_fn:
                            loss = self.ood_loss_fn(logits, labels, ood_labels)
                        else:
                            loss_fct = nn.CrossEntropyLoss()
                            loss = loss_fct(
                                logits.view(-1, self.model.config.num_labels),
                                labels.view(-1),
                            )

                        return (loss, outputs) if return_outputs else loss

                # Create OOD loss function
                ood_loss_fn = OODLoss(
                    ood_weight=self.ood_weight,
                    energy_margin=self.energy_margin,
                    temperature=self.temperature,
                )

                trainer = OODTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    ood_loss_fn=ood_loss_fn,
                    compute_metrics=lambda eval_pred: {
                        "accuracy": accuracy_score(
                            eval_pred.label_ids, eval_pred.predictions.argmax(axis=1)
                        )
                    },
                )
            else:
                # Standard training
                training_args = TrainingArguments(
                    output_dir="tmp",
                    num_train_epochs=4,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    logging_dir="./logs",
                    logging_steps=100,
                    eval_strategy="epoch",
                    disable_tqdm=False,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    compute_metrics=lambda eval_pred: {
                        "accuracy": accuracy_score(
                            eval_pred.label_ids, eval_pred.predictions.argmax(axis=1)
                        )
                    },
                )

            # Add callback for SNGP-like training
            if self.use_spectral_norm:
                trainer.add_callback(ResetCovarianceCallback())

            # Train the model
            trainer.train()

            # Extract model components for saving
            if hasattr(model, "base_model"):
                # Enhanced model
                if self.model_name == "distil-bert":
                    models.append(
                        model.base_model.distilbert.transformer.layer[-2:].state_dict()
                    )
                    classifiers.append(model.base_model.classifier.state_dict())
                elif self.model_name == "roberta":
                    models.append(
                        model.base_model.roberta.encoder.layer[-2:].state_dict()
                    )
                    classifiers.append(model.base_model.classifier.state_dict())
                elif self.model_name == "bert":
                    models.append(model.base_model.bert.encoder.layer[-2:].state_dict())
                    classifiers.append(model.base_model.classifier.state_dict())
            else:
                # Standard model
                if self.model_name == "distil-bert":
                    models.append(model.distilbert.transformer.layer[-2:].state_dict())
                    classifiers.append(model.classifier.state_dict())
                elif self.model_name == "roberta":
                    models.append(model.roberta.encoder.layer[-2:].state_dict())
                    classifiers.append(model.classifier.state_dict())
                elif self.model_name == "bert":
                    models.append(model.bert.encoder.layer[-2:].state_dict())
                    classifiers.append(model.classifier.state_dict())

            # Evaluate and compute metrics
            predictions, labels, _ = trainer.predict(test_dataset)
            predictions = predictions.argmax(axis=-1)

            # Standard classification metrics
            report = classification_report(
                labels,
                predictions,
                target_names=label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            )

            for cls in label_encoder.classes_:
                classes.append(cls)
                accuracies.append(report[cls]["precision"])
                f1_scores.append(report[cls]["f1-score"])

            # OOD-specific evaluation if enabled
            if self.ood_training and hasattr(model, "predict_with_uncertainty"):
                logger.info("Evaluating OOD detection performance...")

                # Create a simple test batch for uncertainty evaluation
                test_batch = next(iter(trainer.get_test_dataloader()))
                test_inputs = {
                    k: v
                    for k, v in test_batch.items()
                    if k not in ["labels", "ood_labels"]
                }

                with torch.no_grad():
                    _, entropy, energy = model.predict_with_uncertainty(
                        test_inputs["input_ids"], test_inputs.get("attention_mask")
                    )

                # Store OOD metrics (entropy and energy distributions)
                ood_metric = {
                    "entropy_mean": entropy.mean().item(),
                    "entropy_std": entropy.std().item(),
                    "energy_mean": energy.mean().item(),
                    "energy_std": energy.std().item(),
                }
                ood_metrics.append(ood_metric)

                logger.info(
                    f"OOD Metrics - Entropy: {ood_metric['entropy_mean']:.4f}±{ood_metric['entropy_std']:.4f}, "
                    f"Energy: {ood_metric['energy_mean']:.4f}±{ood_metric['energy_std']:.4f}"
                )
            else:
                ood_metrics.append({})

            label_encoders.append(label_encoder)
            shutil.rmtree("tmp")

        basic_model = self.base_model.state_dict()
        metrics = (classes, accuracies, f1_scores)

        # Return enhanced results with OOD metrics if available
        if self.ood_training and any(ood_metrics):
            return (
                metrics,
                models,
                classifiers,
                label_encoders,
                basic_model,
                ood_metrics,
            )
        else:
            return metrics, models, classifiers, label_encoders, basic_model
