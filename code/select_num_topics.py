import os

import artm
import click
import warnings

from tqdm import tqdm

from bigartm_tools import convert_to_batch, transform_batch, create_dictionary, config_artm_logs
from build_tm_model import build_model, BigARTM
from datasets.imdb import IMDB
from transform_predictor import Predictor
from artm import BatchVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
import time
import pandas as pd

from datasets import AGNews, R8, R52, Ohsumed, NG20, DBPedia

warnings.filterwarnings("ignore")
config_artm_logs()


@click.command()
@click.option('--data_folder', default='data', help='Path to folder with dataset')
@click.option('--report_folder', default='data/reports', help='Path to folder to save reports')
def select_params(data_folder, report_folder):

    datasets = [R8, R52, NG20, Ohsumed, IMDB]

    for dataset in datasets:
        data = dataset(data_folder, lemmatize=True, exclude_stop_words=True, exclude_label_field=False)
        t = time.time()
        train_batch = convert_to_batch(data.train_docs)
        test_batch = convert_to_batch(data.test_docs)
        dictionary = create_dictionary(train_batch)
        df = pd.DataFrame()

        for topic_num in tqdm(range(10, 250, 15)):
            model = build_model(topic_num)
            class_ids = dict(zip(data.columns, [1] * len(data.columns)))
            class_ids['label'] = 5

            model.class_ids = class_ids
            model.initialize(dictionary.filter(min_df=10))
            model.fit_offline(batch_vectorizer=BatchVectorizer(batches=train_batch, process_in_memory_model=model),
                              num_collection_passes=3)
            train_x = transform_batch(model, train_batch)
            test_x = transform_batch(model, test_batch)

            svm = LinearSVC()
            svm.fit(train_x, data.train_labels)

            predictor = Predictor(topic_num)
            predictor.fit(train_x, data.train_labels)

            t = time.time() - t
            f1_train_svm = precision_recall_fscore_support(data.train_labels, svm.predict(train_x), average='macro')[2]
            f1_test_svm = precision_recall_fscore_support(data.test_labels, svm.predict(test_x), average='macro')[2]

            f1_train_tm = precision_recall_fscore_support(data.train_labels, predictor.predict(train_x), average='macro')[2]
            f1_test_tm = precision_recall_fscore_support(data.test_labels, predictor.predict(test_x), average='macro')[2]

            df = df.append({'topic_num': topic_num, 'f1_train_svm': f1_train_svm, 'f1_test_svm': f1_test_svm,
                        'f1_train_tm': f1_train_tm, 'f1_test_tm': f1_test_tm}, ignore_index=True)
        df.to_csv(os.path.join(report_folder, f'{data.name}.csv'))


if __name__ == '__main__':
    select_params()
