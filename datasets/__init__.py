from typing import Union, List, Optional
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer
from transformers import RobertaTokenizer

from keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import tokenize_and_preserve_labels, tokenize_and_preserve_labels_sentencepiece
from utils.preprocessing import SentenceGetter, split_snt_in_chunks


class TransformerFromConllDFDataModule(pl.LightningDataModule):
    def __init__(
            self, train_path, val_path, test_path, tokenizer, max_token_seq_len=32, max_word_seq_len=20,
            train_batch_size=64, eval_batch_size=64, sample_perc=1.0):
        super(TransformerFromConllDFDataModule, self).__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self.tokenizer = tokenizer

        self.max_token_seq_len = max_token_seq_len
        self.max_word_seq_len = max_word_seq_len

        self.batch_train_size = train_batch_size
        self.batch_eval_size = eval_batch_size

        self.sample_perc = sample_perc

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.tag2idx, self.idx2tag = {}, {}
        self.n_classes = -1

        # load the data
        '''
        self.train_df = self.get_dataframe(self.train_path, sample_perc=self.sample_perc)
        self.val_df = self.get_dataframe(self.val_path, sample_perc=self.sample_perc)
        self.test_df = self.get_dataframe(self.test_path, sample_perc=self.sample_perc)
        '''

        #'''
        # load the data only for test
        self.train_df = self.get_dataframe(self.train_path, sample_perc=self.sample_perc).head(100)
        self.val_df = self.get_dataframe(self.val_path, sample_perc=self.sample_perc).head(100)
        self.test_df = self.get_dataframe(self.test_path, sample_perc=self.sample_perc).head(100)
        #'''

        # overfitting for DEBUG fixme
        # self.train_df = self.train_df.iloc[10000:self.train_df.shape[0]//100]
        # self.val_df = self.train_df
        # self.test_df = self.train_df



        # create class dictionary
        self.tag2idx = self.get_dict_from_column((self.train_df, self.val_df, self.test_df), column="Tag")
        self.idx2tag = dict(zip(self.tag2idx.values(), self.tag2idx.keys()))
        self.n_classes = len(self.tag2idx)

        self.docid2idx = self.get_dict_from_column((self.train_df, self.val_df, self.test_df), column="doc_id")
        self.idx2docid = dict(zip(self.docid2idx.values(), self.docid2idx.keys()))

    def setup(self, stage: Optional[str] = None) -> None:
        # sets train, val and test sets
        if stage in (None, "fit"):
            self.train_set = self._preprocess_for_transformers(self.train_df, self.tokenizer)
            self.val_set = self._preprocess_for_transformers(self.val_df, self.tokenizer)

        # if stage in (None, "test", "predict"):
        self.test_set = self._preprocess_for_transformers(self.test_df, self.tokenizer)

    @staticmethod
    def get_dict_from_column(partition_dfs, column="Tag"):
        # build a dictionary from column values
        tags_sets = [set(partition_df[column].values) for partition_df in partition_dfs]
        tag_values = set()
        tag_values = tag_values.union(*tags_sets)

        return {t: i for i, t in enumerate(tag_values)}

    @staticmethod
    def get_dataframe(df_path, sample_perc=1.0):
        df = pd.read_csv(df_path, encoding="utf-8", sep="\t", doublequote=False, quoting=3).fillna(method="ffill")

        if 0.0 < sample_perc < 1.0:
            print("--- Warning sampling of DF is random and breaks the sentences use just to make code debugging ---")
            # doc_ids = list(df["doc_id"].unique())[:50]
            # sentence_ids = list(range(50))
            # df = df[(df["doc_id"].isin(doc_ids)) & (df["Sentence #"].isin(sentence_ids))]
            df = df.sample(frac=sample_perc)

        return df

    def _preprocess_for_transformers(self, df, tokenizer):
        sentences_partition = []
        getter_partition = SentenceGetter(df)
        for sentence in getter_partition.sentences:
            o = split_snt_in_chunks(sentence, self.max_word_seq_len)
            sentences_partition.extend(o)

        sentences = [[s[0] for s in sentence] for sentence in sentences_partition]
        labels_partition = [[s[1] for s in sentence] for sentence in sentences_partition]
        doc_ids_partition = [[s[2] for s in sentence] for sentence in sentences_partition]

        if isinstance(tokenizer, RobertaTokenizer):
            raw_data = [tokenize_and_preserve_labels_sentencepiece(sent, labs, doc_id, tokenizer) for sent, labs, doc_id in zip(sentences, labels_partition, doc_ids_partition)]

        elif isinstance(tokenizer, BertTokenizer):
            raw_data = [tokenize_and_preserve_labels(sent, labs, doc_id, tokenizer) for sent, labs, doc_id in zip(sentences, labels_partition, doc_ids_partition)]
        else:
            print(f"Error - {tokenizer} not recognized")
            return NotImplementedError

        tokenized_texts, labels, docs = zip(*raw_data)

        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=self.max_token_seq_len, dtype="long", value=0,
                                  truncating="post", padding="post")

        labels_ids = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in labels],
                                   maxlen=self.max_token_seq_len, value=-100, padding="post",
                                   dtype="long", truncating="post")

        docs_ids = pad_sequences([[self.docid2idx.get(doc_id) for doc_id in doc_ids] for doc_ids in docs],
                                 maxlen=self.max_token_seq_len, value=-100, padding="post",
                                 dtype="long", truncating="post")

        return input_ids, labels_ids, docs_ids

    @staticmethod
    def make_dataloader(data, batch_size, sampler_cls="rand"):
        inputs_ids, labels_ids, docs_ids = data
        tr_inputs = torch.tensor(inputs_ids)
        tr_labels = torch.tensor(labels_ids, dtype=torch.long)

        attention_masks = [[float(i != 0) for i in ii] for ii in inputs_ids]
        tr_masks = torch.tensor(attention_masks)

        tr_docs = torch.tensor(docs_ids)

        dataset = TensorDataset(tr_inputs, tr_masks, tr_labels, tr_docs)
        sampler = RandomSampler(dataset) if sampler_cls == "rand" else SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers= 3, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.train_set, self.batch_train_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.make_dataloader(self.val_set, self.batch_eval_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.make_dataloader(self.test_set, self.batch_eval_size, sampler_cls='seq')
