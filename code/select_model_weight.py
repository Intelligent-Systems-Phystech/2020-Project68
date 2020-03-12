import itertools
import json
import os

import artm
import warnings

import click
from tqdm import tqdm

from bigartm_tools import convert_to_batch, transform_batch, create_dictionary, config_artm_logs
from build_tm_model import build_model, BigARTM
from transform_predictor import Predictor
from artm import BatchVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
import time
import pandas as pd

from datasets import DATASETS


def get_all_combinations(d):
    keys = d.keys()
    values = (d[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations


warnings.filterwarnings("ignore")
config_artm_logs()


@click.command()
@click.option('--data_folder', default='data', help='Path to folder with dataset')
@click.option('--report_folder', default='reports', help='Path to folder to save reports')
@click.option('--datasets', default='20NG', help='Dataset names, slitted by comma (,)')
@click.option('--best_params', default='best_params.json', help='Dataset names, slitted by comma (,)')
def select_weights(data_folder, report_folder, datasets, best_params):

    datasets = datasets.split(',')
    if not all(d in DATASETS for d in datasets):
        raise ValueError('Not all datasets found')

    for dataset in datasets:
        data = DATASETS[dataset](data_folder, lemmatize=True, exclude_stop_words=True, exclude_label_field=False)
        t = time.time()
        train_batch = convert_to_batch(data.train_docs)
        test_batch = convert_to_batch(data.test_docs)
        dictionary = create_dictionary(train_batch)
        df = pd.DataFrame()

        model_params = {'topic_num': 100}
        if os.path.exists(best_params):
            with open(best_params) as f:
                model_params = json.load(f)[dataset]
            print('Model params', model_params)

        class_weights = [0.5, 1, 2, 3, 5, 10]
        params = dict(zip(data.columns, [class_weights] * len(data.columns)))
        print('Grid search params', params)

        for class_ids in tqdm(get_all_combinations(params)):
            model = build_model(**model_params)

            model.class_ids = class_ids
            model.initialize(dictionary.filter(min_df=10))
            model.fit_offline(batch_vectorizer=BatchVectorizer(batches=train_batch, process_in_memory_model=model),
                              num_collection_passes=3)
            train_x = transform_batch(model, train_batch)
            test_x = transform_batch(model, test_batch)

            svm = LinearSVC()
            svm.fit(train_x, data.train_labels)

            predictor = Predictor(**model_params)
            predictor.fit(train_x, data.train_labels)

            t = time.time() - t
            f1_train_svm = precision_recall_fscore_support(data.train_labels, svm.predict(train_x), average='macro')[2]
            f1_test_svm = precision_recall_fscore_support(data.test_labels, svm.predict(test_x), average='macro')[2]

            f1_train_tm = precision_recall_fscore_support(data.train_labels, predictor.predict(train_x), average='macro')[2]
            f1_test_tm = precision_recall_fscore_support(data.test_labels, predictor.predict(test_x), average='macro')[2]

            metrics = {'f1_train_svm': f1_train_svm, 'f1_test_svm': f1_test_svm,
                            'f1_train_tm': f1_train_tm, 'f1_test_tm': f1_test_tm}
            metrics.update(class_ids)
            df = df.append(metrics, ignore_index=True)

        df.to_csv(os.path.join(report_folder, f'{data.name}_class_ids.csv'))


if __name__ == '__main__':
    select_weights()