import itertools

import artm
import warnings

from tqdm import tqdm

from bigartm_tools import convert_to_batch, transform_batch, create_dictionary, config_artm_logs
from build_tm_model import build_model, BigARTM
from transform_predictor import Predictor
from artm import BatchVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
import time
import pandas as pd

from datasets import AGNews, R8, R52, Ohsumed, NG20, DBPedia


def get_all_combinations(d):
    keys = d.keys()
    values = (d[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations


warnings.filterwarnings("ignore")
config_artm_logs()

data_folder = 'data'

datasets = [R8, R52, NG20, Ohsumed, IMDB]

for dataset in datasets:
    data = dataset(data_folder, lemmatize=True, exclude_stop_words=True, exclude_label_field=False)
    t = time.time()
    train_batch = convert_to_batch(data.train_docs)
    test_batch = convert_to_batch(data.test_docs)
    dictionary = create_dictionary(train_batch)
    df = pd.DataFrame()

    topic_num = 100  # количество смысловых тем, в которых находятся важные слова
    background_topic_num = 3  # количество "фоновых" тем, в которые мы будем помещать бессмысленные слова (стоп-слова)
    document_passes_num = 10  # количество проходов по документу внутри одного E-шага
    processors_num = 12

    class_weights = [0.5, 1, 2, 3, 5, 10]
    params = dict(zip(data.columns, [class_weights] * len(data.columns)))

    for class_ids in tqdm(get_all_combinations(params)):
        model = build_model(topic_num)

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

        metrics = {'f1_train_svm': f1_train_svm, 'f1_test_svm': f1_test_svm,
                        'f1_train_tm': f1_train_tm, 'f1_test_tm': f1_test_tm}
        metrics.update(class_ids)
        df = df.append(metrics, ignore_index=True)

    df.to_excel(f'data/reports/{dataset.__name__}_class_ids.xlsx')
