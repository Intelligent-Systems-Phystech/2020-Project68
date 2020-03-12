# pip install git+https://github.com/latorrefabian/topmine.git
import bz2

from topmine import Corpus, TopmineTokenizer
import pandas as pd


class TopMine:
    def __init__(self, threshold: float = 0.1, min_support: int = 1, ngrams_only=True):
        self.ngrams_only = ngrams_only
        self.min_support = min_support
        self.threshold = threshold
        self.tokenizer: TopmineTokenizer = None

    def fit(self, X, y=None):
        corpus = Corpus(documents=X)
        self.tokenizer = TopmineTokenizer(threshold=self.threshold, min_support=self.min_support)
        self.tokenizer.fit(corpus=corpus)

    def transform(self, X, y=None):
        ngrams = []
        for doc in X:
            doc_ngrams = self.tokenizer.transform_document(doc)
            doc_ngrams = [x.replace(' ', '_') for x in doc_ngrams]
            if self.ngrams_only:
                doc_ngrams = filter(lambda x: '_' in x, doc_ngrams)
            ngrams.append(' '.join(doc_ngrams))
        return ngrams


if __name__ == '__main__':
    with bz2.BZ2File('data/r8/lemmatized_wo_stopwords/train.bz2', "r") as f:
        df_train = pd.read_json(f, lines=True)

    ngammer = TopMine()
    ngammer.fit(df_train['text'])
    print(ngammer.transform(df_train['text'].head()))

