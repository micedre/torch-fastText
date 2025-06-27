"""torchTextClassifiers: A Unified Framework for Text Classification

This package provides a comprehensive, extensible framework for building and training
text classification models. It offers a unified API that abstracts away the complexity
of different model architectures while providing flexibility for advanced users.

Package Overview:
=================

torchTextClassifiers is designed around a modular architecture that separates concerns:

1. **Core Framework** (`torchTextClassifiers.py`):
   - Unified classifier interface
   - Factory pattern for model creation
   - Configuration management
   - Training orchestration

2. **Classifier Implementations** (`classifiers/`):
   - FastText: Fast and efficient text classification
   - Extensible for additional architectures (BERT, CNN, LSTM, etc.)

3. **Utilities** (`utilities/`):
   - Text preprocessing
   - Input validation
   - Helper functions

Architecture Diagram:
=====================

    ┌─────────────────────────────────────────────────────┐
    │                User Interface                       │
    │  create_fasttext(), build(), train(), predict()    │
    └─────────────────────┬───────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────┐
    │              torchTextClassifiers                   │
    │         (Main Classifier Interface)                 │
    │                                                     │
    │  ┌─────────────────┐    ┌─────────────────────────┐│
    │  │ ClassifierType  │    │  ClassifierFactory      ││
    │  │   (Enum)        │    │    (Registry)           ││
    │  └─────────────────┘    └─────────────────────────┘│
    └─────────────────────┬───────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────┐
    │           Classifier Implementations                │
    │                                                     │
    │  ┌─────────────────────────────────────────────────┐│
    │  │              FastText                           ││
    │  │                                                 ││
    │  │ ┌─────────────┐ ┌──────────────┐ ┌────────────┐││
    │  │ │   Config    │ │   Wrapper    │ │   Model    │││
    │  │ │             │ │              │ │            │││
    │  │ └─────────────┘ └──────────────┘ └────────────┘││
    │  │                       │                        ││
    │  │ ┌─────────────┐ ┌──────▼──────┐ ┌────────────┐││
    │  │ │ Tokenizer   │ │  Lightning  │ │  Dataset   │││
    │  │ │             │ │   Module    │ │            │││
    │  │ └─────────────┘ └─────────────┘ └────────────┘││
    │  └─────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────┘

Key Features:
=============

**Unified API**: Consistent interface across different model types
- Same methods for building, training, and prediction
- Standardized configuration management
- Uniform data handling

**PyTorch Lightning Integration**: Production-ready training
- Automatic GPU/CPU handling
- Built-in logging and checkpointing
- Early stopping and learning rate scheduling
- Distributed training support

**Flexible Data Support**: Handle various input formats
- Text-only classification
- Mixed text + categorical features
- Batch processing
- Variable-length sequences

**Extensible Architecture**: Easy to add new models
- Plugin-based classifier registration
- Abstract base classes for consistency
- Modular component design

**Performance Optimized**: Efficient implementation
- Sparse embeddings for memory efficiency
- Parallel tokenization
- Optimized data loading

Currently Supported Models:
===========================

**FastText Classifier**:
- Character and word n-gram features
- Sparse embedding support
- Mixed feature support (text + categorical)
- Fast training and inference
- Memory efficient for large vocabularies

**Future Extensions** (Framework Ready):
- BERT-based classifiers
- CNN text classifiers  
- LSTM/GRU classifiers
- Transformer architectures

Quick Start Guide:
==================

Basic Text Classification:
--------------------------

    >>> from torchTextClassifiers import create_fasttext
    >>> import numpy as np
    >>> 
    >>> # Create classifier
    >>> classifier = create_fasttext(
    ...     embedding_dim=100,
    ...     sparse=False,
    ...     num_tokens=10000,
    ...     min_count=2,
    ...     min_n=3,
    ...     max_n=6,
    ...     len_word_ngrams=2,
    ...     num_classes=2
    ... )
    >>> 
    >>> # Prepare data
    >>> X_train = np.array(["positive text", "negative text"])
    >>> y_train = np.array([1, 0])
    >>> X_val = np.array(["validation text"])
    >>> y_val = np.array([1])
    >>> 
    >>> # Build and train
    >>> classifier.build(X_train, y_train)
    >>> classifier.train(X_train, y_train, X_val, y_val, num_epochs=10, batch_size=32)
    >>> 
    >>> # Predict
    >>> predictions = classifier.predict(np.array(["new text sample"]))

Mixed Features Classification:
------------------------------

    >>> # Text + categorical data
    >>> X_mixed = np.array([
    ...     ["Product is great", 1, 0],  # text + category1 + category2
    ...     ["Poor quality", 0, 1]
    ... ])
    >>> y_train = np.array([1, 0])
    >>> 
    >>> # Create classifier with categorical support
    >>> classifier = create_fasttext(
    ...     embedding_dim=50,
    ...     sparse=False,
    ...     num_tokens=5000,
    ...     min_count=1,
    ...     min_n=3,
    ...     max_n=6,
    ...     len_word_ngrams=2,
    ...     num_classes=2,
    ...     categorical_vocabulary_sizes=[2, 2],  # vocab size for each category
    ...     categorical_embedding_dims=10,
    ...     num_categorical_features=2
    ... )

Advanced Configuration:
=======================

The framework supports extensive customization through configuration objects:

    >>> from torchTextClassifiers.classifiers.fasttext import FastTextConfig, FastTextWrapper
    >>> from torchTextClassifiers import torchTextClassifiers, ClassifierType
    >>> 
    >>> # Custom configuration
    >>> config = FastTextConfig(
    ...     embedding_dim=300,
    ...     sparse=True,  # For memory efficiency
    ...     num_tokens=100000,
    ...     min_count=5,
    ...     min_n=3,
    ...     max_n=6,
    ...     len_word_ngrams=2,
    ...     learning_rate=0.01,
    ...     # ... other parameters
    ... )
    >>> 
    >>> # Create classifier with custom config
    >>> classifier = torchTextClassifiers(ClassifierType.FASTTEXT, config)

Error Handling and Validation:
==============================

The framework includes comprehensive input validation and error handling:

- **Data Validation**: Automatic checks for data format and consistency
- **Configuration Validation**: Parameter range and compatibility checks  
- **Runtime Safety**: Graceful handling of edge cases and errors
- **Informative Messages**: Clear error messages with suggested fixes

Performance Guidelines:
=======================

**Memory Optimization**:
- Use sparse=True for large vocabularies (>50k tokens)
- Reduce embedding_dim for faster training
- Use categorical features instead of text when possible

**Training Speed**:
- Increase batch_size on GPUs
- Use multiple workers for data loading
- Enable mixed precision training

**Model Quality**:
- Increase min_count to reduce noise
- Tune n-gram ranges based on text characteristics
- Use validation data for early stopping

This framework provides a solid foundation for text classification tasks while
remaining flexible enough to support advanced use cases and custom extensions.
"""

from .torchTextClassifiers import torchTextClassifiers, ClassifierType, ClassifierFactory

# Convenience imports for FastText
try:
    from .classifiers.fasttext.fasttext import FastTextFactory
    
    # Expose FastText convenience methods at package level for easy access
    create_fasttext = FastTextFactory.create_fasttext
    build_fasttext_from_tokenizer = FastTextFactory.build_from_tokenizer
    
except ImportError:
    # FastText module not available - define placeholder functions
    def create_fasttext(*args, **kwargs):
        raise ImportError("FastText module not available")
    
    def build_fasttext_from_tokenizer(*args, **kwargs):
        raise ImportError("FastText module not available")

__all__ = [
    "torchTextClassifiers",
    "ClassifierType", 
    "ClassifierFactory",
    "create_fasttext",
    "build_fasttext_from_tokenizer",
]

__version__ = "1.0.0"