"""torchTextClassifiers package."""

from .torchTextClassifiers import torchTextClassifiers, ClassifierType, ClassifierFactory

# Convenience imports for FastText
try:
    from .classifiers.fasttext.factory import FastTextFactory
    
    # Expose FastText convenience methods at package level
    create_fasttext = FastTextFactory.create_fasttext
    build_fasttext_from_tokenizer = FastTextFactory.build_from_tokenizer
    
except ImportError:
    # FastText module not available
    pass

__all__ = [
    "torchTextClassifiers",
    "ClassifierType", 
    "ClassifierFactory",
    "create_fasttext",
    "build_fasttext_from_tokenizer",
]