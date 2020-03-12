import os

from .common import ReportedResults
from .base_dataset import JsonDataset


class R52(JsonDataset):
    def __init__(self, data_folder, ngrams=True, lemmatize=True, exclude_stop_words=True,
                 exclude_label_field=True):
        self.reported_results = ReportedResults()
        self.name = "R52"

        if not lemmatize:
            path = os.path.join(data_folder, f'{self.name.lower()}/original_text/')
        elif not exclude_stop_words:
            path = os.path.join(data_folder, f'{self.name.lower()}/lemmatized/')
        else:
            path = os.path.join(data_folder, f'{self.name.lower()}/lemmatized_wo_stopwords/')
        super().__init__(path, ngrams, exclude_label_field)