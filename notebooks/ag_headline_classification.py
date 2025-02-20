import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchFastText import torchFastText
import numpy as np

def merge_cat(cat):
    if cat in ['World', 'Top News', 'Europe', 'Italia', 'U.S.', 'Top Stories']:
        return 'World News'
    if cat in ['Sci/Tech', 'Software and Developement', 'Toons', 'Health', 'Music Feeds']:
        return 'Tech and Stuff'
    return cat


if __name__ == "__main__":

    news_df = pd.read_parquet('data/ag_news_full.parquet')
    news_df['category_final'] = news_df['category'].apply(lambda x: merge_cat(x))
    print(news_df['category_final'].value_counts())

    cat_encoder = LabelEncoder()

    news_df['cat'] = cat_encoder.fit_transform(news_df['category_final'])
    news_df['title_headline'] = news_df['title'] + ' '  + news_df['description']

    news_train, news_test = train_test_split(news_df, stratify=news_df['cat'], test_size=0.5, shuffle=True, random_state=42)

    X = news_train['title_headline']
    y = news_train['cat']
    print(X.count())
    print(y.count())
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, random_state=42)

    torchft_model = torchFastText( 
        num_tokens=2000,
        embedding_dim=10,
        min_count=2,
        min_n=2,
        max_n=3,
        len_word_ngrams=4,
        sparse=False,
    )
    
    # Train the model
    torchft_model.train(
          np.asarray(X_train),
          np.asarray(y_train),
          np.asarray(X_test),
          np.asarray(y_test),
          lr=0.1,
          num_epochs=2,
          num_workers=2,
          batch_size=32,
          trainer_params={'enable_progress_bar': True, 'profiler': 'simple'}
    )

    
    predictions,_ = torchft_model.predict(np.asarray(news_test['title_headline']))
    predictions_decoded = cat_encoder.inverse_transform(predictions.reshape(-1))
    print (f"Accuracy : {(predictions_decoded.reshape(-1) == news_test['category_final']).mean():0.2%}")
