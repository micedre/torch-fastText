import torchFastText
import unittest
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent

class tftTest(unittest.TestCase):
      
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        test_data_path = source_dir.as_posix() + '/test_data/imdb_titles_100000.csv'
        df_sample = pd.read_csv(test_data_path)
        self.texts = np.asarray(df_sample.filter(['primaryTitle','isAdult']))
        self.labels = np.asarray(df_sample['genres'])
        num_buckets = 20
        embedding_dim = 100
        min_count=1
        min_n=3
        max_n=6
        len_word_ngrams=10
        sparse=False 
        self.torchfasttext = torchFastText.torchFastText(num_buckets=num_buckets,embedding_dim=embedding_dim,min_count=min_count,min_n=min_n,max_n=max_n,len_word_ngrams=len_word_ngrams,sparse=sparse)



    def test_init(self):
        num_buckets = 20
        embedding_dim = 100
        min_count=1
        min_n=3
        max_n=6
        len_word_ngrams=10
        sparse=False 
        self.torchfasttext = torchFastText.torchFastText(num_buckets=num_buckets,embedding_dim=embedding_dim,min_count=min_count,min_n=min_n,max_n=max_n,len_word_ngrams=len_word_ngrams,sparse=sparse)
        self.assertTrue(True,  msg="Initialisation OK")
        
    def test_train_no_categorical_variables(self):
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.texts, self.labels, test_size=0.2, stratify=self.labels
        )
        self.torchfasttext.train(np.asarray(train_texts), np.asarray(train_labels), np.asarray(val_texts),np.asarray(val_labels), num_epochs=5, batch_size=64, lr=0.001, num_workers=4)
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()