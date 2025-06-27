"""FastText classifier implementation.

This module consolidates the core FastText components and wrapper functionality:
- Configuration dataclass
- Loss functions  
- Factory methods for creating classifiers
- High-level wrapper interface

Consolidates what was previously in core.py and wrapper.py.
"""

from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from ..base import BaseClassifierConfig, BaseClassifierWrapper
from typing import Optional, List, TYPE_CHECKING, Union, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .tokenizer import NGramTokenizer
from .model import FastTextModel, FastTextModule, FastTextModelDataset
from ...utilities.checkers import check_X, check_Y
import logging
from torch.optim import SGD, Adam

if TYPE_CHECKING:
    from ...torchTextClassifiers import torchTextClassifiers, ClassifierType

logger = logging.getLogger()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FastTextConfig(BaseClassifierConfig):
    """Configuration for FastText classifier."""
    # Embedding matrix
    embedding_dim: int
    sparse: bool

    # Tokenizer-related
    num_tokens: int
    min_count: int
    min_n: int
    max_n: int
    len_word_ngrams: int

    # Optional parameters
    num_classes: Optional[int] = None
    num_rows: Optional[int] = None

    # Categorical variables
    categorical_vocabulary_sizes: Optional[List[int]] = None
    categorical_embedding_dims: Optional[Union[List[int], int]] = None
    num_categorical_features: Optional[int] = None

    # Model-specific parameters
    direct_bagging: Optional[bool] = True
    
    # Training parameters
    learning_rate: float = 4e-3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FastTextConfig":
        return cls(**data)


# ============================================================================
# Loss Functions
# ============================================================================

class OneVsAllLoss(nn.Module):
    def __init__(self):
        super(OneVsAllLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Compute One-vs-All loss

        Args:
            logits: Tensor of shape (batch_size, num_classes) containing classification scores
            targets: Tensor of shape (batch_size) containing true class indices

        Returns:
            loss: Mean loss value across the batch
        """

        num_classes = logits.size(1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # For each sample, treat the true class as positive and all others as negative
        # Using binary cross entropy for each class
        loss = F.binary_cross_entropy_with_logits(
            logits,  # Raw logits
            targets_one_hot,  # Target probabilities
            reduction="none",  # Don't reduce yet to allow for custom weighting if needed
        )

        # Sum losses across all classes for each sample, then take mean across batch
        return loss.sum(dim=1).mean()


# ============================================================================
# Wrapper Implementation
# ============================================================================

class FastTextWrapper(BaseClassifierWrapper):
    """Wrapper for FastText classifier."""
    
    def __init__(self, config: FastTextConfig):
        super().__init__(config)
        self.config: FastTextConfig = config
    
    def build_tokenizer(self, training_text: np.ndarray) -> None:
        """Build NGram tokenizer for FastText."""
        self.tokenizer = NGramTokenizer(
            self.config.min_count,
            self.config.min_n,
            self.config.max_n,
            self.config.num_tokens,
            self.config.len_word_ngrams,
            training_text,
        )
    
    def _build_pytorch_model(self) -> None:
        """Build FastText PyTorch model."""
        if self.config.num_rows is None:
            if self.tokenizer is None:
                raise ValueError(
                    "Please provide a tokenizer or num_rows."
                )
            else:
                self.config.num_rows = self.tokenizer.padding_index + 1
        else:
            if self.tokenizer is not None:
                if self.config.num_rows != self.tokenizer.padding_index + 1:
                    logger.warning(
                        f"Divergent values for num_rows: {self.config.num_rows} and {self.tokenizer.padding_index + 1}. "
                        f"Using max value."
                    )
                self.config.num_rows = max(self.config.num_rows, self.tokenizer.padding_index + 1)
        
        self.padding_idx = self.config.num_rows - 1
        
        # Update tokenizer padding index if necessary
        if self.tokenizer is not None and self.padding_idx != self.tokenizer.padding_index:
            self.tokenizer.padding_index = self.padding_idx
        
        self.pytorch_model = FastTextModel(
            tokenizer=self.tokenizer,
            embedding_dim=self.config.embedding_dim,
            num_rows=self.config.num_rows,
            num_classes=self.config.num_classes,
            categorical_vocabulary_sizes=self.config.categorical_vocabulary_sizes,
            categorical_embedding_dims=self.config.categorical_embedding_dims,
            padding_idx=self.padding_idx,
            sparse=self.config.sparse,
            direct_bagging=self.config.direct_bagging,
        )
    
    def _check_and_init_lightning(
        self,
        optimizer=None,
        optimizer_params=None,
        lr=None,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=None,
        patience_scheduler=3,
        loss=torch.nn.CrossEntropyLoss(),
    ) -> None:
        """Initialize Lightning module for FastText."""
        if optimizer is None:
            if lr is None:
                lr = getattr(self.config, 'learning_rate', 4e-3)  # Use config or default
            self.optimizer = SGD if self.config.sparse else Adam
            self.optimizer_params = {"lr": lr}
        else:
            self.optimizer = optimizer
            if optimizer_params is None:
                if lr is not None:
                    self.optimizer_params = {"lr": lr}
                else:
                    logger.warning("No optimizer parameters provided. Using defaults.")
                    self.optimizer_params = {}
        
        self.scheduler = scheduler
        
        if scheduler_params is None:
            logger.warning("No scheduler parameters provided. Using defaults.")
            self.scheduler_params = {
                "mode": "min",
                "patience": patience_scheduler,
            }
        else:
            self.scheduler_params = scheduler_params
        
        self.loss = loss
        
        self.lightning_module = FastTextModule(
            model=self.pytorch_model,
            loss=self.loss,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            scheduler=self.scheduler,
            scheduler_params=self.scheduler_params,
            scheduler_interval="epoch",
        )
    
    def predict(self, X: np.ndarray, top_k=1, preprocess=False, verbose=False) -> np.ndarray:
        """Make predictions with FastText model."""
        if not self.trained:
            raise Exception("Model must be trained first.")
        
        text, categorical_variables, no_cat_var = check_X(X)
        if categorical_variables is not None:
            if categorical_variables.shape[1] != self.config.num_categorical_features:
                raise Exception(
                    f"X must have the same number of categorical variables as training data."
                )
        else:
            assert self.pytorch_model.no_cat_var == True
        
        predictions, confidence = self.pytorch_model.predict(
            text, categorical_variables, top_k=top_k, preprocess=preprocess
        )
        
        # Return just predictions, squeeze out the top_k dimension if top_k=1
        if top_k == 1:
            predictions = predictions.squeeze(-1)
        
        # Convert to numpy array for consistency
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
            
        return predictions
    
    def validate(self, X: np.ndarray, Y: np.ndarray, batch_size=256, num_workers=12) -> float:
        """Validate FastText model."""
        if not self.trained:
            raise Exception("Model must be trained first.")
        
        # Use predict method which handles input validation and returns just predictions
        predictions = self.predict(X)
        y = check_Y(Y)
        
        # Convert predictions to numpy if it's a tensor
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        
        # Calculate accuracy
        accuracy = (predictions == y).mean()
        return float(accuracy)
    
    def predict_and_explain(self, X: np.ndarray, top_k=1):
        """Predict and explain with FastText model."""
        if not self.trained:
            raise Exception("Model must be trained first.")
        
        text, categorical_variables, no_cat_var = check_X(X)
        if categorical_variables is not None:
            if categorical_variables.shape[1] != self.config.num_categorical_features:
                raise Exception(
                    f"X must have the same number of categorical variables as training data ({self.config.num_categorical_features})."
                )
        else:
            assert self.pytorch_model.no_cat_var == True
        
        return self.pytorch_model.predict_and_explain(text, categorical_variables, top_k=top_k)
    
    def create_dataset(self, texts: np.ndarray, labels: np.ndarray, categorical_variables: np.ndarray = None):
        """Create FastText dataset."""
        return FastTextModelDataset(
            categorical_variables=categorical_variables,
            texts=texts,
            outputs=labels,
            tokenizer=self.tokenizer,
        )
    
    def create_dataloader(self, dataset, batch_size: int, num_workers: int = 0, shuffle: bool = True):
        """Create FastText dataloader."""
        return dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    
    def load_best_model(self, checkpoint_path: str) -> None:
        """Load best FastText model from checkpoint."""
        self.lightning_module = FastTextModule.load_from_checkpoint(
            checkpoint_path,
            model=self.pytorch_model,
            loss=self.loss,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            scheduler=self.scheduler,
            scheduler_params=self.scheduler_params,
            scheduler_interval="epoch",
        )
        self.pytorch_model = self.lightning_module.model.to("cpu")
        self.trained = True
        self.pytorch_model.eval()


# ============================================================================
# Factory Methods
# ============================================================================

class FastTextFactory:
    """Factory class for creating FastText classifiers with convenience methods.
    
    This factory provides static methods for creating FastText classifiers with
    common configurations. It handles the complexities of configuration creation
    and classifier initialization, offering a simplified API for users.
    
    All methods return fully initialized torchTextClassifiers instances that are
    ready for building and training.
    """
    
    @staticmethod
    def create_fasttext(
        embedding_dim: int,
        sparse: bool,
        num_tokens: int,
        min_count: int,
        min_n: int,
        max_n: int,
        len_word_ngrams: int,
        **kwargs
    ) -> "torchTextClassifiers":
        """Create a FastText classifier with the specified configuration.
        
        This is the primary method for creating FastText classifiers. It creates
        a configuration object with the provided parameters and initializes a
        complete classifier instance.
        
        Args:
            embedding_dim: Dimension of word embeddings
            sparse: Whether to use sparse embeddings
            num_tokens: Maximum number of tokens in vocabulary
            min_count: Minimum count for tokens to be included in vocabulary
            min_n: Minimum length of character n-grams
            max_n: Maximum length of character n-grams
            len_word_ngrams: Length of word n-grams to use
            **kwargs: Additional configuration parameters (e.g., num_classes,
                     categorical_vocabulary_sizes, etc.)
                     
        Returns:
            torchTextClassifiers: Initialized FastText classifier instance
            
        Example:
            >>> from torchTextClassifiers.classifiers.fasttext.fasttext import FastTextFactory
            >>> classifier = FastTextFactory.create_fasttext(
            ...     embedding_dim=100,
            ...     sparse=False,
            ...     num_tokens=10000,
            ...     min_count=2,
            ...     min_n=3,
            ...     max_n=6,
            ...     len_word_ngrams=2,
            ...     num_classes=3
            ... )
        """
        from ...torchTextClassifiers import torchTextClassifiers, ClassifierType
        
        config = FastTextConfig(
            embedding_dim=embedding_dim,
            sparse=sparse,
            num_tokens=num_tokens,
            min_count=min_count,
            min_n=min_n,
            max_n=max_n,
            len_word_ngrams=len_word_ngrams,
            **kwargs
        )
        return torchTextClassifiers(ClassifierType.FASTTEXT, config)
    
    @staticmethod
    def build_from_tokenizer(
        tokenizer,  # NGramTokenizer
        embedding_dim: int,
        num_classes: Optional[int],
        categorical_vocabulary_sizes: Optional[List[int]] = None,
        sparse: bool = False,
        **kwargs
    ) -> "torchTextClassifiers":
        """Create FastText classifier from an existing trained tokenizer.
        
        This method is useful when you have a pre-trained tokenizer and want to
        create a classifier that uses the same vocabulary and tokenization scheme.
        The resulting classifier will have its tokenizer and model architecture
        pre-built.
        
        Args:
            tokenizer: Pre-trained NGramTokenizer instance
            embedding_dim: Dimension of word embeddings
            num_classes: Number of output classes
            categorical_vocabulary_sizes: Sizes of categorical feature vocabularies
            sparse: Whether to use sparse embeddings
            **kwargs: Additional configuration parameters
            
        Returns:
            torchTextClassifiers: Classifier with pre-built tokenizer and model
            
        Raises:
            ValueError: If the tokenizer is missing required attributes
            
        Example:
            >>> # Assume you have a pre-trained tokenizer
            >>> classifier = FastTextFactory.build_from_tokenizer(
            ...     tokenizer=my_tokenizer,
            ...     embedding_dim=100,
            ...     num_classes=2,
            ...     sparse=False
            ... )
            >>> # The classifier is ready for training without building
            >>> classifier.train(X_train, y_train, X_val, y_val, ...)
        """
        from ...torchTextClassifiers import torchTextClassifiers, ClassifierType
        
        # Ensure the tokenizer has required attributes
        required_attrs = ["min_count", "min_n", "max_n", "num_tokens", "word_ngrams"]
        if not all(hasattr(tokenizer, attr) for attr in required_attrs):
            missing_attrs = [attr for attr in required_attrs if not hasattr(tokenizer, attr)]
            raise ValueError(f"Missing attributes in tokenizer: {missing_attrs}")
        
        config = FastTextConfig(
            num_tokens=tokenizer.num_tokens,
            embedding_dim=embedding_dim,
            min_count=tokenizer.min_count,
            min_n=tokenizer.min_n,
            max_n=tokenizer.max_n,
            len_word_ngrams=tokenizer.word_ngrams,
            sparse=sparse,
            num_classes=num_classes,
            categorical_vocabulary_sizes=categorical_vocabulary_sizes,
            **kwargs
        )
        
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, config)
        classifier.classifier_wrapper.tokenizer = tokenizer
        classifier.classifier_wrapper._build_pytorch_model()
        
        return classifier
    
    @staticmethod
    def from_dict(config_dict: dict) -> FastTextConfig:
        """Create FastText configuration from dictionary.
        
        This method is used internally by the configuration factory system
        to recreate FastText configurations from serialized data.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            FastTextConfig: Reconstructed configuration object
            
        Example:
            >>> config_dict = {
            ...     'embedding_dim': 100,
            ...     'num_tokens': 5000,
            ...     'min_count': 1,
            ...     # ... other parameters
            ... }
            >>> config = FastTextFactory.from_dict(config_dict)
        """
        return FastTextConfig.from_dict(config_dict)