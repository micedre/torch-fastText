"""FastText classifier factory and convenience methods."""

from typing import Optional, List, TYPE_CHECKING
import numpy as np

from .config import FastTextConfig
from .wrapper import FastTextWrapper
from .tokenizer import NGramTokenizer

if TYPE_CHECKING:
    from ...torchTextClassifiers import torchTextClassifiers, ClassifierType


class FastTextFactory:
    """Factory class for creating FastText classifiers with convenience methods."""
    
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
        """Convenience method to create FastText classifier."""
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
        tokenizer: NGramTokenizer,
        embedding_dim: int,
        num_classes: Optional[int],
        categorical_vocabulary_sizes: Optional[List[int]] = None,
        sparse: bool = False,
        **kwargs
    ) -> "torchTextClassifiers":
        """Build FastText classifier from existing tokenizer."""
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
        """Create FastText config from dictionary."""
        return FastTextConfig.from_dict(config_dict)