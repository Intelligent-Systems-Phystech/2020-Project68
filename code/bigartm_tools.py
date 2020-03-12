import os
import artm
import uuid
from collections import Counter
from artm import BatchVectorizer, Dictionary, messages


def extract_unique_words(documents):
    words = set()
    for d in documents:
        for m in d:
            for w in d[m].split():
                words.add(w)
    return words


def convert_to_batch(documents, index=0, size=1000, dictionary=None):
    if type(documents) is not list:
        documents = [documents]
    if len(documents) > size:
        chunks = [documents[x:x + size] for x in range(0, len(documents), size)]
        return sum([convert_to_batch(x, i + index, dictionary=dictionary) for i, x in enumerate(chunks)], [])
    batch = artm.messages.Batch()
    batch.id = str(uuid.uuid4())

    # add all unique tokens (and their class_id) to batch.token and batch.class_id
    unique_tokens = {}
    for doc in documents:
        for column in doc.keys():
            if column not in unique_tokens:
                unique_tokens[column] = set()
            for w in doc[column].split():
                if dictionary is None or w in dictionary:
                    unique_tokens[column].add(w)

    ids = {}

    for class_id in sorted(unique_tokens):
        for token in sorted(unique_tokens[class_id]):
            ids[token + class_id] = len(ids)
            batch.token.append(token)
            batch.class_id.append(class_id)

    for i, doc in enumerate(documents):
        item = batch.item.add()
        item.title = str(size*index + i)
        item.id = size*index + i
        for column in doc:
            c = Counter(doc[column].split())
            for w in c:
                if dictionary is None or w in dictionary:
                    item.token_id.append(ids[w + column])    # token_id refers to an index in batch.token
                    item.token_weight.append(c[w])

    return [batch]


def transform_batch(model, batch):
    batch_vectorizer = BatchVectorizer(batches=batch, process_in_memory_model=model)
    df = model.transform(batch_vectorizer=batch_vectorizer)
    return df.sort_index(axis=1).T.to_numpy()


def transform_batch_label(model, batch, label_col):
    batch_vectorizer = BatchVectorizer(batches=batch, process_in_memory_model=model)
    p_cd_test = model.transform(batch_vectorizer=batch_vectorizer, predict_class_id=label_col).T
    y_pred = p_cd_test.idxmax(axis=1).astype(int)
    return y_pred

def transform_batch_vectorizer(model, batch):
    df = model.transform(batch)
    return df.sort_index(axis=1).T.to_numpy()


def get_pwt(model, batch):
    batch_vectorizer = BatchVectorizer(batches=batch, process_in_memory_model=model)
    df = model.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type='dense_ptdw')
    df.columns = [batch[0].token[i] for i in batch[0].item[0].token_id]
    return df.T


def create_dictionary(batches):
    tf = Counter()
    df = Counter()
    for batch in batches:
        for item in batch.item:
            for freq, tid in zip(item.token_weight, item.token_id):
                cls, word = batch.class_id[tid], batch.token[tid]
                key = cls + "::" + word
                tf[key] += freq
                df[key] += 1
    global_n = sum(tf.values())
    dictionary = Dictionary()
    dictionary_data = messages.DictionaryData()
    dictionary_data.name = uuid.uuid1().urn.replace(':', '')
    # dictionary_data.
    for key in df.keys():
        cls, word = key.split('::')
        dictionary_data.token.append(word)
        dictionary_data.class_id.append(cls)
        dictionary_data.token_tf.append(tf[key])
        dictionary_data.token_df.append(df[key])
        dictionary_data.token_value.append(1.)

    dictionary.create(dictionary_data)
    return dictionary


def config_artm_logs():
    lc = artm.messages.ConfigureLoggingArgs()

    log_folder = './logs'
    os.makedirs(log_folder, exist_ok=True)
    lc.log_dir = log_folder
    
    lib = artm.wrapper.LibArtm(logging_config=lc)
    lc.minloglevel = 0  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
    lib.ArtmConfigureLogging(lc)

