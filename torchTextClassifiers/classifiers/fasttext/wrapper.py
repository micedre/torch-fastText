from ..base import BaseClassifierWrapper
from .config import FastTextConfig
from .tokenizer import NGramTokenizer
from .pytorch_model import FastTextModel
from .lightning_module import FastTextModule
from .dataset import FastTextModelDataset
from ...utilities.checkers import check_X, check_Y
import logging
import numpy as np
import torch
from torch.optim import SGD, Adam

logger = logging.getLogger()


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
                raise ValueError("Please provide a learning rate")
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
        
        return self.pytorch_model.predict(
            text, categorical_variables, top_k=top_k, preprocess=preprocess
        )
    
    def validate(self, X: np.ndarray, Y: np.ndarray, batch_size=256, num_workers=12) -> float:
        """Validate FastText model."""
        if not self.trained:
            raise Exception("Model must be trained first.")
        
        text, categorical_variables, no_cat_var = check_X(X)
        y = check_Y(Y)
        
        if categorical_variables is not None:
            if categorical_variables.shape[1] != self.config.num_categorical_features:
                raise Exception(
                    f"X must have the same number of categorical variables as training data ({self.config.num_categorical_features})"
                )
        else:
            assert self.pytorch_model.no_cat_var == True
        
        dataset = FastTextModelDataset(
            categorical_variables=categorical_variables,
            texts=text,
            outputs=y,
            tokenizer=self.tokenizer,
        )
        dataloader = dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers)
        
        return self.trainer.test(self.pytorch_model, test_dataloaders=dataloader, verbose=False)
    
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

