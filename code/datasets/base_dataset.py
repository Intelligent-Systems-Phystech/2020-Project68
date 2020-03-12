import bz2
import os
import pandas as pd

from datasets.topmine_transformer import TopMine


class JsonDataset:
    def __init__(self, path, ngrams=True, exclude_label_field=True):
        self.train_docs = []
        self.train_labels = []
        self.test_docs = []
        self.test_labels = []
        self.ngrammer = None
        self.columns = []

        with bz2.BZ2File(os.path.join(path, 'train.bz2'), "r") as f:
            df_train = pd.read_json(f, lines=True)
            df_train = self._cast_to_str(df_train)
            df_train['label'] = df_train['label'].astype(str)
            if ngrams:
                self.ngrammer = TopMine()
                self.ngrammer.fit(df_train["text"])
                df_train["ngrams"] = self.ngrammer.transform(df_train["text"])
            self.train_labels = df_train['label'].to_list()
            if exclude_label_field:
                del df_train["label"]
            self.columns = df_train.columns.to_list()
            self.train_docs = df_train.T.to_dict().values()  # dataframe to list of dicts

        with bz2.BZ2File(os.path.join(path, 'test.bz2'), "r") as f:
            df_test = pd.read_json(f, lines=True)
            df_test = self._cast_to_str(df_test)
            df_test['label'] = df_test['label'].astype(str)
            if ngrams:
                self.ngrammer = TopMine()
                self.ngrammer.fit(df_test["text"])
                df_test["ngrams"] = self.ngrammer.transform(df_test["text"])
            self.test_labels = df_test['label'].to_list()
            del df_test["label"]
            self.test_docs = df_test.T.to_dict().values()

    def _cast_to_str(self, df):
        for col in df.columns:
            df[col] = df[col].astype(str)
        return df
