import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


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
    chunks = []
    new_chunk = []
    for i in range(len(sentence)):
        new_chunk.append(sentence[i])
        if len(new_chunk) == chunk_len or i == len(sentence)-1:
            chunks.append(new_chunk)
            new_chunk = []

    return chunks


def tokenize_and_preserve_labels_sentencepiece(sentence, text_labels, doc_ids, tokenizer):
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
    sentences_partition = []
    getter_partition = SentenceGetter(df)
    for sentence in getter_partition.sentences:
        o = split_snt_in_chunks(sentence, MAX_CHUNKS_LEN)
        sentences_partition.extend(o)

    sentences = [[s[0] for s in sentence] for sentence in sentences_partition]
    labels_partition = [[s[1] for s in sentence] for sentence in sentences_partition]
    doc_ids_partition = [[s[2] for s in sentence] for sentence in sentences_partition]

    # raw_train = [tokenize_and_preserve_labels(sent, labs, doc_id, tokenizer) for sent, labs, doc_id in zip(sentences, labels_partition, doc_ids_partition)]
    raw_train = [tokenize_and_preserve_labels_sentencepiece(sent, labs, doc_id, tokenizer) for sent, labs, doc_id in zip(sentences, labels_partition, doc_ids_partition)]
    tokenized_texts_train = [input_tuple[0] for input_tuple in raw_train]
    labels_train = [input_tuple[1] for input_tuple in raw_train]
    docs_train = [input_tuple[2] for input_tuple in raw_train]
    return tokenized_texts_train, labels_train, docs_train


def make_dataloader(inputs_ids, labels_ids, batch_size):
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
