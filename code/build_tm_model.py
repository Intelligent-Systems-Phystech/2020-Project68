import artm
from artm import BatchVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin

from bigartm_tools import convert_to_batch, create_dictionary, transform_batch_label, transform_batch
from transform_predictor import Predictor


def build_model(topic_num=100, background_topic_num=3, document_passes_num=10, processors_num=12):
    # количество смысловых тем, в которых находятся важные слова
    # количество "фоновых" тем, в которые мы будем помещать бессмысленные слова (стоп-слова)
    # количество проходов по документу внутри одного E-шага

    topics_names = ["subject_" + str(i) for i in range(topic_num)] + \
                   ["background_" + str(i) for i in range(background_topic_num)]  # назначаем имена темам

    subj_topics = topics_names[:topic_num]
    bgr_topics = topics_names[topic_num:]

    model = artm.ARTM(num_document_passes=document_passes_num,
                      num_topics=topic_num + background_topic_num,
                      topic_names=topics_names,
                      seed=100,  # helps to get stable results
                      num_processors=processors_num)

    model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='Decorrelator', tau=10 ** 4))  # обычный декоррелятор
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SmoothTheta',
                                                             topic_names=bgr_topics,
                                                             tau=0.3))  # сглаживаем Theta для фоновых тем
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta',
                                                             topic_names=subj_topics,
                                                             tau=-0.3))  # разреживаем Theta для "хороших" тем
    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SmoothPhi',
                                                           topic_names=bgr_topics,
                                                           class_ids=["text"],
                                                           tau=0.1))  # сглаживаем Theta для фоновых тем
    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi',
                                                           topic_names=subj_topics,
                                                           class_ids=["text"],
                                                           tau=-0.1))  # разреживаем Theta для "хороших" тем
    #    model.regularizers.add(artm.LabelRegularizationPhiRegularizer(class_ids=["label"]))    # этот регуляризатор мало у кого дает
    #    # хороший результат, но ты попробуй :) у меня он вылетает с ошибкой :(

    return model


class BigARTM(BaseEstimator, ClassifierMixin):
    def __init__(self, topic_num=100, background_topic_num=3, document_passes_num=10, processors_num=12, class_ids = None,
                label_col=None):
        if class_ids is not None:
            self.class_ids = class_ids
        else:
            self.class_ids = {}
        self.processors_num = processors_num
        self.document_passes_num = document_passes_num
        self.background_topic_num = background_topic_num
        self.topic_num = topic_num
        self.label_col = label_col

        topics_names = ["subject_" + str(i) for i in range(topic_num)] + \
                       ["background_" + str(i) for i in range(background_topic_num)]  # назначаем имена темам

        subj_topics = topics_names[:topic_num]
        bgr_topics = topics_names[topic_num:]

        self.model = artm.ARTM(num_document_passes=document_passes_num,
                          num_topics=topic_num + background_topic_num,
                          topic_names=topics_names,
                          seed=100,  # helps to get stable results
                          num_processors=processors_num)

        self.model.regularizers.add(
            artm.DecorrelatorPhiRegularizer(name='Decorrelator', tau=10 ** 4))  # обычный декоррелятор
        self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SmoothTheta',
                                                                 topic_names=bgr_topics,
                                                                 tau=0.3))  # сглаживаем Theta для фоновых тем
        self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta',
                                                                 topic_names=subj_topics,
                                                                 tau=-0.3))  # разреживаем Theta для "хороших" тем
        self.model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SmoothPhi',
                                                               topic_names=bgr_topics,
                                                               class_ids=["text"],
                                                               tau=0.1))  # сглаживаем Theta для фоновых тем
        self.model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi',
                                                               topic_names=subj_topics,
                                                               class_ids=["text"],
                                                               tau=-0.1))  # разреживаем Theta для "хороших" тем
        self.model.class_ids = self.class_ids

    def fit(self, X, y=None):
        train_batch = convert_to_batch(X)
        dictionary = create_dictionary(train_batch)
        self.model.initialize(dictionary.filter(min_df=10))
        self.model.fit_offline(batch_vectorizer=BatchVectorizer(batches=train_batch, process_in_memory_model=self.model),
                          num_collection_passes=3)

    def predict(self, X, y=None):
        test_batch = convert_to_batch(X)
        train_x = transform_batch(self.model, test_batch)


        predictor = Predictor(self.topic_num)
        predictor.fit(train_x, data.train_labels)
        batch_vectorizer = BatchVectorizer(batches=test_batch, process_in_memory_model=self.model)
        p_cd_test = self.model.transform(batch_vectorizer=batch_vectorizer, predict_class_id=self.label_col).T
        y_pred = p_cd_test.idxmax(axis=1).astype(int)
        return y_pred
