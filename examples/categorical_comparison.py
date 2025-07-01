"""
Categorical Features Comparison Example

This example demonstrates the performance difference between:
1. A FastText classifier using only text features
2. A FastText classifier using both text and categorical features

Based on the notebook example using the same SIRENE dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers import create_fasttext
from torchTextClassifiers.utilities.preprocess import clean_text_feature
import time

def categorize_surface(
    df: pd.DataFrame, surface_feature_name: str, like_sirene_3: bool = True
) -> pd.DataFrame:
    """
    Categorize the surface of the activity.
    """
    df_copy = df.copy()
    df_copy[surface_feature_name] = df_copy[surface_feature_name].replace("nan", np.nan)
    df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(float)
    
    if surface_feature_name not in df.columns:
        raise ValueError(f"Surface feature {surface_feature_name} not found in DataFrame.")
    
    if not (pd.api.types.is_float_dtype(df_copy[surface_feature_name])):
        raise ValueError(f"Surface feature {surface_feature_name} must be a float variable.")

    if like_sirene_3:
        df_copy["surf_cat"] = pd.cut(
            df_copy[surface_feature_name],
            bins=[0, 120, 400, 2500, np.inf],
            labels=["1", "2", "3", "4"],
        ).astype(str)
    else:
        df_copy["surf_log"] = np.log(df[surface_feature_name])
        df_copy["surf_cat"] = pd.cut(
            df_copy.surf_log,
            bins=[0, 3, 4, 5, 12],
            labels=["1", "2", "3", "4"],
        ).astype(str)

    df_copy[surface_feature_name] = df_copy["surf_cat"].replace("nan", "0")
    df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(int)
    df_copy = df_copy.drop(columns=["surf_log", "surf_cat"], errors="ignore")
    return df_copy

def clean_and_tokenize_df(
    df,
    categorical_features=["EVT", "CJ", "NAT", "TYP", "CRT"],
    text_feature="libelle_processed",
    label_col="apet_finale",
):
    df.fillna("nan", inplace=True)

    df = df.rename(
        columns={
            "evenement_type": "EVT",
            "cj": "CJ",
            "activ_nat_et": "NAT",
            "liasse_type": "TYP",
            "activ_surf_et": "SRF",
            "activ_perm_et": "CRT",
        }
    )

    les = []
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        les.append(le)

    df = categorize_surface(df, "SRF", like_sirene_3=True)
    df = df[[text_feature, "EVT", "CJ", "NAT", "TYP", "SRF", "CRT", label_col]]

    return df, les

def add_libelles(
    df: pd.DataFrame,
    df_naf: pd.DataFrame,
    y: str,
    text_feature: str,
    textual_features: list,
    categorical_features: list,
):
    missing_codes = set(df_naf["code"])
    fake_obs = df_naf[df_naf["code"].isin(missing_codes)]
    fake_obs[y] = fake_obs["code"]
    fake_obs[text_feature] = fake_obs[[text_feature]].apply(
        lambda row: " ".join(f"[{col}] {val}" for col, val in row.items() if val != ""), axis=1
    )
    df = pd.concat([df, fake_obs[[col for col in fake_obs.columns if col in df.columns]]])

    if textual_features is not None:
        for feature in textual_features:
            df[feature] = df[feature].fillna(value="")
    if categorical_features is not None:
        for feature in categorical_features:
            df[feature] = df[feature].fillna(value="NaN")

    print(f"\t*** {len(missing_codes)} codes have been added in the database...")
    return df

def stratified_split_rare_labels(X, y, test_size=0.2, min_train_samples=1):
    # Get unique labels and their frequencies
    unique_labels, label_counts = np.unique(y, return_counts=True)

    # Separate rare and common labels
    rare_labels = unique_labels[label_counts == 1]

    # Create initial mask for rare labels to go into training set
    rare_label_mask = np.isin(y, rare_labels)

    # Separate data into rare and common label datasets
    X_rare = X[rare_label_mask]
    y_rare = y[rare_label_mask]
    X_common = X[~rare_label_mask]
    y_common = y[~rare_label_mask]

    # Split common labels stratified
    X_common_train, X_common_test, y_common_train, y_common_test = train_test_split(
        X_common, y_common, test_size=test_size, stratify=y_common
    )

    # Combine rare labels with common labels split
    X_train = np.concatenate([X_rare, X_common_train])
    y_train = np.concatenate([y_rare, y_common_train])
    X_test = X_common_test
    y_test = y_common_test

    return X_train, X_test, y_train, y_test


def load_and_prepare_data():
    """Load and prepare the same data as used in the notebook"""
    print("📊 Using Sirene dataset sample for demonstration...")
    df = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-ape/extractions/20241027_sirene4.parquet")
    #df = df.sample(10000, random_state=42)  # Smaller sample to avoid disk space issues
    print(f"✅ Loaded {len(df)} samples from SIRENE dataset")
   
    categorical_features = ["evenement_type", "cj",  "activ_nat_et", "liasse_type", "activ_surf_et", "activ_perm_et"]
    text_feature = "libelle"
    y = "apet_finale"
    textual_features = None

    naf2008 = pd.read_csv("https://minio.lab.sspcloud.fr/projet-ape/data/naf2008.csv", sep=";")
    df = add_libelles(df, naf2008, y, text_feature, textual_features, categorical_features)

    df["libelle_processed"] = clean_text_feature(df["libelle"])
    df, _ = clean_and_tokenize_df(df, text_feature="libelle_processed")

    X_text_only = df[['libelle_processed']].values
    X_mixed = df[['libelle_processed', 'EVT', 'CJ', 'NAT', 'TYP', 'CRT', 'SRF']].values
    y = df['apet_finale'].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return X_text_only, X_mixed, y, encoder
    

def create_fallback_dataset():
    """Create a synthetic dataset as fallback"""
    print("📊 Creating synthetic dataset...")
    
    # Simulate SIRENE-like data
    business_types = ['retail', 'restaurant', 'service', 'manufacturing', 'tech']
    regions = [0, 1, 2, 3, 4]  # Encoded regions
    legal_forms = [0, 1, 2, 3]  # Encoded legal forms
    
    n_samples = 2000
    data = []
    
    for i in range(n_samples):
        # Generate business description
        business_type = np.random.choice(business_types)
        if business_type == 'retail':
            descriptions = ['magasin de vêtements', 'commerce de chaussures', 'vente d\'électronique']
            sectors = [0, 0, 1]
        elif business_type == 'restaurant':
            descriptions = ['restaurant italien', 'fast food', 'café']
            sectors = [2, 2, 2]
        elif business_type == 'service':
            descriptions = ['coiffure', 'réparation automobile', 'conseil']
            sectors = [3, 4, 5]
        elif business_type == 'manufacturing':
            descriptions = ['fabrication métallique', 'production textile']
            sectors = [6, 7]
        else:  # tech
            descriptions = ['développement logiciel', 'services informatiques']
            sectors = [8, 8]
        
        idx = np.random.randint(len(descriptions))
        description = descriptions[idx]
        sector = sectors[idx]
        
        data.append({
            'libelle_processed': description,
            'EVT': np.random.choice([0, 1, 2]),  # Event type
            'CJ': np.random.choice(legal_forms),  # Legal form
            'NAT': np.random.choice([0, 1]),  # Nature
            'TYP': np.random.choice([0, 1, 2]),  # Type
            'CRT': np.random.choice([0, 1]),  # Creation
            'SRF': np.random.choice([0, 1, 2, 3]),  # Surface
            'apet_finale': sector
        })
    
    df = pd.DataFrame(data)
    
    X_text_only = df[['libelle_processed']].values
    X_mixed = df[['libelle_processed', 'EVT', 'CJ', 'NAT', 'TYP', 'CRT', 'SRF']].values
    y = df['apet_finale'].values
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    print(f"✅ Created synthetic dataset with {len(df)} samples")
    print(f"   - {len(np.unique(y))} classes")
    
    return X_text_only, X_mixed, y, encoder

def train_and_evaluate_model(X, y, model_name, use_categorical=False):
    """Train and evaluate a FastText model"""
    print(f"\n🎯 Training {model_name}...")
    
  
    # Split data into train/validation/test using stratified_split_rare_labels twice
    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = stratified_split_rare_labels(
        X, y, test_size=0.2
    )
    print(f"X_Test: {X_test[:10]}")
    # Second split: separate validation from remaining data (10% of original = 12.5% of remaining)
    val_size_adjusted = 0.1 / 0.8  # 10% of original data / 80% remaining data
    X_train, X_val, y_train, y_val = stratified_split_rare_labels(
        X_temp, y_temp, test_size=val_size_adjusted
    )
    
    # Model parameters
    if use_categorical:
        # For mixed model - get vocabulary sizes from data
        cat_data = X_train[:, 1:]  # Categorical features
        vocab_sizes = (np.max(cat_data, axis=0) + 1).astype(int).tolist()
        
        model_params = {
            "embedding_dim": 50,
            "sparse": False,
            "num_tokens": 10000,
            "min_count": 1,
            "min_n": 3,
            "max_n": 6,
            "len_word_ngrams": 2,
            "categorical_embedding_dims": 10,
            #"num_categorical_features": num_cat_var,
            "categorical_vocabulary_sizes": vocab_sizes,
            #"categorical_embedding_dims": 10
        }
        print(f"   Categorical vocabulary sizes: {vocab_sizes}")
    else:
        # For text-only model
        model_params = {
            "embedding_dim": 50,
            "sparse": False,
            "num_tokens": 100000,
            "min_count": 1,
            "min_n": 3,
            "max_n": 6,
            "len_word_ngrams": 2
        }
    
    # Training parameters - reduced to save disk space
    train_params = {
        "num_epochs": 10,
        "batch_size": 256,
        "patience_train": 3,
        "lr": 0.004,
        "verbose": True
    }

    extra_trainer_params = {
        "enable_progress_bar": False,
        
    }
    
    # Create and build model
    start_time = time.time()
    
    classifier = create_fasttext(**model_params)
    classifier.build(X_train, y_train)
    
    # Train model - disable logging to save disk space
    classifier.train(
        X_train, y_train, X_val, y_val, **train_params,
        trainer_params=extra_trainer_params
    )
    
    training_time = time.time() - start_time
    
    # Handle predictions based on model type - use separate test set for validation
    if use_categorical:
        # Skip validation for mixed model due to categorical prediction bug
        print("   ✅ Running validation for text-with-categorical-variables model...")
        try:
            # Validate on separate test set (not used for training)
            test_accuracy = classifier.validate(X_test, y_test)
            predictions = classifier.predict(X_test)
            print(f"   Test accuracy (separate validation): {test_accuracy:.3f}")
        except Exception as e:
            print(f"   ⚠️  Validation failed: {e}")
            test_accuracy = 0.0
            predictions = np.zeros(len(y_test))
    else:
        # Text-only model works fine for predictions
        print("   ✅ Running validation for text-only model...")
        try:
            # Validate on separate test set (not used for training)
            test_accuracy = classifier.validate(X_test, y_test)
            predictions = classifier.predict(X_test)
            print(f"   Test accuracy (separate validation): {test_accuracy:.3f}")
        except Exception as e:
            print(f"   ⚠️  Validation failed: {e}")
            test_accuracy = 0.0
            predictions = np.zeros(len(y_test))
    
    return {
        'model_name': model_name,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'predictions': predictions,
        'y_test': y_test,
        'classifier': classifier
    }

def analyze_predictions(results_text_only, results_mixed, encoder=None, sample_texts=None):
    """Analyze and compare predictions between models"""
    print("\n📈 Prediction Analysis:")
    print("=" * 60)
    
    # Check if we have valid predictions from text-only model
    has_text_predictions = results_text_only['test_accuracy'] > 0
    has_mixed_predictions = results_mixed['test_accuracy'] > 0
    
    if has_text_predictions:
        print("✅ Text-only model predictions available")
        analyze_text_only_predictions(results_text_only, encoder, sample_texts)
    else:
        print("⚠️  Text-only model predictions not available")
    
    if has_mixed_predictions:
        print("\n✅ Mixed model predictions available")
        # This would be implemented when the bug is fixed
        print("   (Mixed model prediction analysis would go here)")
    else:
        print("\n⚠️  Mixed model predictions not available due to categorical bug")
        print("   Expected improvements when bug is fixed:")
        print("   - Better handling of business context")
        print("   - Improved classification for sector-specific patterns")
        print("   - Enhanced performance on companies with specific characteristics")




def main():
    print("🔀 FastText Classifier: Categorical Features Comparison")
    print("=" * 60)
    print("Comparing FastText performance with and without categorical features")
    print("Using the same SIRENE dataset as in the notebook example.")
    print()
    
    # Load and prepare data (same as notebook)
    X_text_only, X_mixed, y, encoder = load_and_prepare_data()
    
    # Train models
    print(f"\n🚀 Training Models:")
    print("-" * 40)
    
    # Text-only model
    results_text_only = train_and_evaluate_model(
        X_text_only, y, "Text-Only FastText", use_categorical=False
    )
    
    # Mixed model (text + categorical)
    results_mixed = train_and_evaluate_model(
        X_mixed, y, "Mixed Features FastText", use_categorical=True
    )
    
    # Compare results
    print(f"\n📊 Results Comparison:")
    print("=" * 50)
    print(f"{'Model':<25} {'Test Acc (Validation)':<20} {'Time (s)':<10}")
    print("-" * 50)
    print(f"{'Text-Only':<25} {results_text_only['test_accuracy']:<20.3f} "
          f"{results_text_only['training_time']:<10.1f}")
    print(f"{'Mixed Features':<25} {results_mixed['test_accuracy']:<20.3f} "
          f"{results_mixed['training_time']:<10.1f}")
    
    # Calculate improvements
    acc_improvement = results_mixed['test_accuracy'] - results_text_only['test_accuracy']
    time_overhead = results_mixed['training_time'] - results_text_only['training_time']
    
    print("-" * 50)
    print(f"Validation Accuracy Improvement: {acc_improvement:+.3f}")
    print(f"Training Time Overhead: {time_overhead:+.1f}s")
       
    print(f"\n🎉 Comparison completed successfully!")

if __name__ == "__main__":
    main()