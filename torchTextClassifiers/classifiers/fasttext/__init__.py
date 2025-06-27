"""FastText Classifier Package

This package provides a complete FastText text classification implementation with PyTorch Lightning
integration. FastText is a fast and efficient text classification method that uses n-gram features
and averaging to create document representations.

Package Structure:
==================

After refactoring, this package contains 3 main files:
- fasttext.py: Configuration, losses, factory methods, and wrapper interface
- tokenizer.py: NGramTokenizer implementation  
- model.py: PyTorch model, Lightning module, and dataset

High-Level Architecture:
========================

    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │  FastTextConfig │    │ FastTextFactory  │    │ FastTextWrapper │
    │  (Configuration)│    │  (Factory)       │    │  (Main API)     │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      FastText Ecosystem       │
                    │                               │
                    │  ┌─────────────────────────┐  │
                    │  │    NGramTokenizer       │  │
                    │  │  (Text Processing)      │  │
                    │  └─────────────────────────┘  │
                    │               │               │
                    │  ┌─────────────▼─────────────┐│
                    │  │      FastTextModel        ││
                    │  │   (PyTorch Model)         ││
                    │  └─────────────────────────────┘│
                    │               │               │
                    │  ┌─────────────▼─────────────┐│
                    │  │    FastTextModule         ││
                    │  │  (Lightning Wrapper)      ││
                    │  └─────────────────────────────┘│
                    └───────────────────────────────────┘

FastText PyTorch Model Architecture:
====================================

The FastTextModel implements the core FastText architecture with support for both
text-only and mixed text+categorical features:

Text-Only Model:
----------------

    Input Text: "Hello world example"
           │
           ▼
    ┌─────────────────────────────────────────────────────┐
    │              NGramTokenizer                         │
    │  "Hello" → [hel, ell, llo] + [Hello] + [wor, ord,   │
    │  "world" → [rld] + [world] + [exa, xam, amp, mpl,   │
    │  "example" → [ple, ple] + [example]                 │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Token IDs: [234, 567, 123, ...])
    ┌─────────────────────────────────────────────────────┐
    │              Embedding Layer                        │
    │  - Learnable embedding matrix: [vocab_size, emb_dim]│
    │  - Maps token IDs to dense vectors                  │
    │  - Supports sparse embeddings for memory efficiency │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Embeddings: [seq_len, emb_dim])
    ┌─────────────────────────────────────────────────────┐
    │              Pooling Layer                          │
    │  - Average pooling across sequence dimension        │
    │  - Result: [batch_size, emb_dim]                    │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Pooled features: [batch_size, emb_dim])
    ┌─────────────────────────────────────────────────────┐
    │           Classification Head                       │
    │  - Linear layer: [emb_dim] → [num_classes]          │
    │  - No activation (logits output)                    │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Logits: [batch_size, num_classes])
        Output Predictions

Mixed Features Model (Text + Categorical):
------------------------------------------

    Input: Text + Categorical Features
           │                    │
           ▼                    ▼
    ┌──────────────┐    ┌─────────────────┐
    │  Text Path   │    │ Categorical Path│
    │  (as above)  │    │                 │
    │      │       │    │ Cat1: Embedding │
    │      ▼       │    │ Cat2: Embedding │
    │   [emb_dim]  │    │ Cat3: Embedding │
    └──────────────┘    │      ...        │
           │            │      │          │
           │            │      ▼          │
           │            │ [sum(cat_dims)] │
           │            └─────────────────┘
           │                    │
           └────────┬───────────┘
                    ▼
            ┌──────────────────┐
            │   Concatenation  │
            │ [emb_dim + cat]  │
            └──────────────────┘
                    │
                    ▼
            ┌──────────────────┐
            │ Classification   │
            │    Head          │
            │ [total] → [cls]  │
            └──────────────────┘
                    │
                    ▼
               Output Logits

Key Model Components:
=====================

1. **Embedding Layer (nn.Embedding)**:
   - Maps token indices to dense vectors
   - Shape: [vocabulary_size, embedding_dim]
   - Supports sparse gradients for memory efficiency
   - Padding tokens are masked during training

2. **Categorical Embeddings (Optional)**:
   - Separate embedding for each categorical feature
   - Each categorical feature gets its own vocabulary
   - Embeddings are summed or concatenated with text features

3. **Pooling Strategy**:
   - Average pooling across sequence dimension
   - Handles variable-length sequences naturally
   - Alternative: Max pooling (not currently implemented)

4. **Classification Head**:
   - Single linear transformation
   - Input: [text_emb_dim + categorical_emb_dim]
   - Output: [num_classes] (raw logits)

Training Process:
=================

1. **Forward Pass**:
   - Tokenize text input using NGramTokenizer
   - Lookup embeddings for tokens and categorical features
   - Apply average pooling to sequence embeddings
   - Concatenate text and categorical embeddings (if present)
   - Pass through classification head to get logits

2. **Loss Computation**:
   - OneVsAllLoss: Binary cross-entropy for each class
   - Standard CrossEntropyLoss: Multi-class classification
   - Supports class weighting for imbalanced datasets

3. **Optimization**:
   - SGD for sparse embeddings (better for large vocabularies)
   - Adam for dense embeddings (faster convergence)
   - Learning rate scheduling with ReduceLROnPlateau

4. **Regularization**:
   - Dropout can be added to embeddings and classification head
   - Early stopping based on validation metrics
   - L2 regularization through weight decay

Mathematical Formulation:
=========================

For text input x = [x₁, x₂, ..., xₙ] and categorical features c = [c₁, c₂, ..., cₘ]:

1. Token Embeddings: E(x) = [e₁, e₂, ..., eₙ] where eᵢ ∈ ℝᵈ
2. Text Representation: h_text = (1/n) ∑ᵢ eᵢ
3. Categorical Embeddings: h_cat = [E_cat₁(c₁), E_cat₂(c₂), ..., E_catₘ(cₘ)]
4. Combined Representation: h = [h_text; h_cat] (concatenation)
5. Output Logits: y = W·h + b where W ∈ ℝᶜˣᵈ, b ∈ ℝᶜ

Performance Characteristics:
============================

- **Memory Efficient**: Sparse embeddings reduce memory usage
- **Fast Training**: Simple architecture allows rapid convergence
- **Scalable**: Handles large vocabularies effectively
- **Flexible**: Supports both text-only and mixed feature scenarios
- **Robust**: Works well with limited training data

Data Flow Summary:
==================

    Raw Text → N-grams → Token IDs → Embeddings → Pooling → Classification → Logits
        ↓         ↓         ↓           ↓          ↓            ↓             ↓
    "hello"   ["hel"]    [42]       [0.1,0.3]   [0.2,0.4]   [2.1,0.5]   Class 0

API Usage Examples:
===================

Basic Usage:
------------

    >>> from torchTextClassifiers.classifiers.fasttext import FastTextFactory
    >>> import numpy as np
    >>> 
    >>> # Create classifier
    >>> classifier = FastTextFactory.create_fasttext(
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
    >>> X_train = np.array(["This is positive", "This is negative"])
    >>> y_train = np.array([1, 0])
    >>> 
    >>> # Build and train
    >>> classifier.build(X_train, y_train)
    >>> classifier.train(X_train, y_train, X_train, y_train, num_epochs=10)
    >>> 
    >>> # Predict
    >>> predictions = classifier.predict(np.array(["New text to classify"]))

Mixed Features Usage:
--------------------

    >>> # Text + categorical features
    >>> X_mixed = np.array([
    ...     ["Product review text", 1, 0, 2],  # text + 3 categorical features
    ...     ["Another review", 0, 1, 1]
    ... ])
    >>> 
    >>> classifier = FastTextFactory.create_fasttext(
    ...     embedding_dim=50,
    ...     sparse=False,
    ...     num_tokens=5000,
    ...     min_count=1,
    ...     min_n=3,
    ...     max_n=6,
    ...     len_word_ngrams=2,
    ...     num_classes=2,
    ...     categorical_vocabulary_sizes=[2, 2, 3],  # vocab sizes for each cat feature
    ...     categorical_embedding_dims=10,  # embedding dim for each cat feature
    ...     num_categorical_features=3
    ... )

Configuration:
==============

All configuration is handled through the FastTextConfig dataclass, which includes:

- **Embedding settings**: dimension, sparsity
- **Tokenizer settings**: n-gram ranges, vocabulary size, minimum counts
- **Model settings**: number of classes, categorical feature support
- **Training settings**: learning rate, optimization strategy

This architecture follows the original FastText paper but with PyTorch implementation,
PyTorch Lightning integration, and additional support for categorical features.
"""

from .fasttext import FastTextConfig, OneVsAllLoss, FastTextFactory, FastTextWrapper
from .tokenizer import NGramTokenizer
from .model import FastTextModel, FastTextModule, FastTextModelDataset

__all__ = [
    "FastTextConfig",
    "OneVsAllLoss", 
    "FastTextFactory",
    "NGramTokenizer",
    "FastTextModel",
    "FastTextModule", 
    "FastTextModelDataset",
    "FastTextWrapper",
]