import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

MAX_CHUNKS_LEN = 40

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t, d) for w, t, d in zip(s["Word"].values.tolist(),
                                                     s["Tag"].values.tolist(), s["doc_id"].values.tolist())]
        self.grouped = self.data.groupby(by=["Sentence #", "doc_id"]).apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def split_snt_in_chunks(sentence, chunk_len):
    """
    A function that split a sentence in chunck of length chunk_len.
    inputs:
    :param sentence: sentence.
    :param chunk_len: length of the chunk.
    :return: chunk.
    """
    chunks = []
    new_chunk = []
    for i in range(len(sentence)):
        new_chunk.append(sentence[i])
        if len(new_chunk) == chunk_len or i == len(sentence)-1:
            chunks.append(new_chunk)
            new_chunk = []
    return chunks


def tokenize_and_preserve_labels_sentencepiece(sentence, text_labels, doc_ids, tokenizer):
    """
    A function that tokenize a sentence preserving its labels
    inputs:
    :param sentence: sentence.
    :param text_labels: labels of words in sentece.
    :param doc_ids: ids of the documento of the sentence.
    :param tokenizer: tokenizer.
    :return:
    tokenized_sentence, the tokenized sentence.
    labels, oredered labels of the sentence.
    preserved_doc_ids, document ids of the sentence.
    """
    labels = []
    preserved_doc_ids = []

    tokenized_sentence = tokenizer.tokenize(' '.join(sentence))

    n_subwords = [0]
    for w in tokenized_sentence:
        if w.startswith("\u0120"):
            n_subwords.append(0)
        n_subwords[-1] += 1

        # if text_labels[curr_w].startswith("B-"):
        #     preserved_labels.append(text_labels[curr_w])

    # e.g. [4,1,2,1,1]...
    for i, label in enumerate(text_labels):
        if label.startswith("B-"):
            preserved_labels = [label] + [label.replace("B-", "I-")] * (n_subwords[i]-1)
        else:
            preserved_labels = [label] * n_subwords[i]
        labels.extend(preserved_labels)
    # ...labels becomes [A, A, A, A, B, C, C, D, E]

    for i, doc_id in enumerate(doc_ids):
        preserved_doc_ids.extend([doc_id] * n_subwords[i])
    return tokenized_sentence, labels, preserved_doc_ids


def tokenize_and_preserve_labels(sentence, text_labels, doc_ids, tokenizer):
    """
    A function that tokenize a sentence preserving its labels. Returns 3 lists of
    tokenized sentece, ordered labels of the tokens, and doc ids of the tokens.
    inputs:
    :param sentence: sentence.
    :param text_labels: labels of words in sentece.
    :param doc_ids: ids of the documento of the sentence.
    :param tokenizer: tokenizer.
    :return:
    tokenized_sentence, the tokenized sentence.
    labels, oredered labels of the sentence.
    preserved_doc_ids, document ids of the sentence.
    """
    tokenized_sentence = []
    labels = []
    preserved_doc_ids = []

    for word, label, doc_id in zip(sentence, text_labels, doc_ids):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        if label.startswith("B-"):
            preserved_labels = [label] + [label.replace("B-", "I-")] * (n_subwords-1)
        else:
            preserved_labels = [label] * n_subwords
        labels.extend(preserved_labels)

        # Preserved docs
        preserved_doc_ids.extend([doc_id] * n_subwords)
    return tokenized_sentence, labels, preserved_doc_ids


def preprocess_for_transformers(df, tokenizer):
    """
    A function that returns 3 lists of tokens, labels and doc ids
    of a tokenize a dataframe.
    inputs:
    :param df: dataframe.
    :param tokenizer: a specif tokenizer of a model.
    :return:
    tokenized_texts, the nested list of tokenized sentences.
    labels, the nested list of labels of the tokenized sentences.
    docs, the nested list of document ids of the tokenized sentences.
    """
    sentences_partition = []
    getter_partition = SentenceGetter(df)
    for sentence in getter_partition.sentences:
        o = split_snt_in_chunks(sentence, MAX_CHUNKS_LEN)
        sentences_partition.extend(o)

    sentences = [[s[0] for s in sentence] for sentence in sentences_partition]
    labels_partition = [[s[1] for s in sentence] for sentence in sentences_partition]
    doc_ids_partition = [[s[2] for s in sentence] for sentence in sentences_partition]

    # raw_train = [tokenize_and_preserve_labels(sent, labs, doc_id, tokenizer)
                   #for sent, labs, doc_id in zip(sentences, labels_partition, doc_ids_partition)]
    raw = [tokenize_and_preserve_labels_sentencepiece(sent, labs, doc_id, tokenizer)
           for sent, labs, doc_id in zip(sentences, labels_partition, doc_ids_partition)]
    tokenized_texts = [input_tuple[0] for input_tuple in raw]
    labels = [input_tuple[1] for input_tuple in raw]
    docs = [input_tuple[2] for input_tuple in raw]
    return tokenized_texts, labels, docs


def make_dataloader(inputs_ids, labels_ids, batch_size):
    """
    A function that returns a DataLoader object made from the dataset of
    inputs ids and labels ids.
    inputs:
    :param inputs_ids: ids of tokenized sentences.
    :param tokenizer: labels of the ids.
    :param bs: batch size.
    :return:
    A dataloader, with random sampler of the dataset dataset of the inputs ids
    and labels ids.
    """
    tr_inputs = torch.tensor(inputs_ids)
    tr_labels = torch.tensor(labels_ids)
    attention_masks = [[float(i != 0.0) for i in ii] for ii in inputs_ids]
    tr_masks = torch.tensor(attention_masks)

    dataset = TensorDataset(tr_inputs, tr_masks, tr_labels)
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def pad_list(l, pad_token, max_l_size, keep_lasts=False, pad_right=True):
    """
    Adds a padding token to a list
    inputs:
    :param l: input list to pad.
    :param pad_token: value to add as padding.
    :param max_l_size: length of the new padded list to return,
    it truncates lists longer that 'max_l_size' without adding
    padding values.
    :param keep_lasts: If True, preserves the max_l_size last elements
    of a sequence (by keeping the same order).  E.g.:
    if keep_lasts is True and max_l_size=3 [1,2,3,4] becomes [2,3,4].
    :param pad_right: If True, default, add pads on the right.
    :return: the list padded or truncated.
    """
    to_pad = []
    max_l = min(max_l_size, len(l))  # maximum len
    l_init = len(l) - max_l if len(l) > max_l and keep_lasts else 0  # initial position where to sample from the list
    l_end = len(l) if len(l) > max_l and keep_lasts else max_l
    for i in range(l_init, l_end):
        to_pad.append(l[i])

    # for j in range(len(l), max_l_size):
    #     to_pad.append(pad_token)
    pad_tokens = [pad_token] * (max_l_size-len(l))
    padded_l = to_pad + pad_tokens if pad_right else pad_tokens + to_pad
    return padded_l

def make_input_ids(tokenized_texts, tokenizer, max_len):
    """
    A function which returns the nasted list of input ids of sentences in
    tokenized texts.
    inputs:
    :param tokenized_texts: input tokenized texts.
    :param tokenizer: a tokenizer.
    :param max_len: maximum length of the texts.
    :return: ids list of the tokenized texts.
    """
    ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype="long", value=0.0,
                              truncating="post", padding="post")
    return ids

def make_tags_list(tags_enum_list, labels, max_len):
    """
    A function which returns the list of tags enumerated of sentences in a
    tokenized text
    inputs:
    :param tag_enum_list: list of tags.
    :param labels: a tokenizer.
    :param max_len: maximum length of the text.
    :return: tags list.
    """
    tags = pad_sequences([[tags_enum_list.get(l) for l in lab] for lab in labels],
                               maxlen=max_len, value=tags_enum_list["PAD"], padding="post",
                               dtype="long", truncating="post")
    return tags

def dataloader(dataframe, tokenizer, tag_enum_list, bs, max_len):
    """
    A function that returns a dataloader object from the dataframe in input. It
    uses the functions make_input_ids and make_tags_list to encodes the dataframe
    and than creates a DataLoader object with the function make_dataloader.
    inputs:
    :param dataframe: A dataframe.
    :param tokenizer: a tokenizer.
    :param tag_enum_list: An enumeration of the dataframe tags.
    :param bs: batch size.
    :param max_len: maximum lenght of a sentece.
    :return: dataloader.
    """
    tokenized_texts, labels, docs = preprocess_for_transformers(dataframe, tokenizer)
    ids = make_input_ids(tokenized_texts, tokenizer, max_len)
    tags = make_tags_list(tag_enum_list, labels, max_len)
    data = make_dataloader(ids, tags, batch_size=bs)
    return data

def dataloader_test(dataframe, tokenizer, tag_enum_list, bs, max_len):
    """
    A function that returns a dataloader object from the dataframe in input. It
    uses the functions make_input_ids and make_tags_list to encodes the dataframe
    and than creates a DataLoader object with the function make_dataloader. (The
    difference from dataloader is that dataloader_test returns also doc_test, a
    list of doc ids).
    inputs:
    :param dataframe: A dataframe.
    :param tokenizer: a tokenizer.
    :param tag_enum_list: An enumeration of the dataframe tags.
    :param bs: batch size.
    :param max_len: maximum lenght of a sentece.
    :return: dataloader.
    """
    tokenized_texts, labels, docs = preprocess_for_transformers(dataframe, tokenizer)
    ids = make_input_ids(tokenized_texts, tokenizer)
    tags = make_tags_list(tag_enum_list, labels)
    data = make_dataloader(ids, tags, batch_size=bs)
    doc_test = [pad_list(doc_test, "[PAD]", max_l_size=MAX_LEN) for doc_test in docs]
    return data, doc_test



def preprocesser(df, tokenizer, tags_enum_list, bs, max_len):
    """
    A function that prepare data for the transformer.
    inputs:
    :param df: a dataframe.
    :param tokenizer: a tokenizer of the selected transformer.
    :param tags_enum_list: enumeration of the dataframe tags.
    :param bs: batch size.
    :param max_len: max length of the tokenized sentence.
    :return: input_ids, tags, dataloader.
    """
    tokenized_texts, labels, docs = preprocess_for_transformers(df, tokenizer)
    ids = make_input_ids(tokenized_texts, tokenizer, max_len)
    tags = make_tags_list(tags_enum_list, labels, max_len)
    dataload = dataloader(df, tokenizer, tags_enum_list, bs, max_len)
    return ids, tags, dataload


