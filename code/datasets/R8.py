import os
from .common import ReportedResults
from .dataset import JsonDataset


class R8(JsonDataset):
    def __init__(self, data_folder, ngrams=True, lemmatize=True, exclude_stop_words=True,
                 exclude_label_field=True):
        self.reported_results = ReportedResults()
        self.name = "R8"

        if not lemmatize:
            path = os.path.join(data_folder, f'{self.name.lower()}/original_text/')
        elif not exclude_stop_words:
            path = os.path.join(data_folder, f'{self.name.lower()}/lemmatized/')
        else:
            path = os.path.join(data_folder, f'{self.name.lower()}/lemmatized_wo_stopwords/')
        super().__init__(path, ngrams, exclude_label_field)
