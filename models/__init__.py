import abc
import torch
import pytorch_lightning as pl
from transformers import AdamW
from seqeval.metrics import f1_score, accuracy_score, recall_score, precision_score


class TransformerModule(pl.LightningModule, metaclass=abc.ABCMeta):
    """
    Compartmental Model abstract class from which other classes should extend
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.lr = kwargs.get("lr", 1e-3)
        # self.optimizer_cls = kwargs.get("optimizer", torch.optim.Adam)
        self.optimizer_cls = kwargs.get("optimizer", AdamW)
        self.model = kwargs.get("model", None)
        self.predictions = []
        self.docs = []
        self.idx2tag = kwargs.get("idx2tag", None)

    def on_epoch_start(self) -> None:
        self.predictions = []
        self.docs = []

    def training_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """

        b_input_ids, b_input_mask, b_labels, _ = batch
        outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)

        # get the loss
        loss = outputs["loss"]

        return loss

    def validation_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        b_input_ids, b_input_mask, b_labels, b_docs = batch
        outputs = self.model(
            b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True
        )

        # compute the actual predicted classes for each token of each sequence in the batch
        logits = outputs["logits"]
        outputs["preds"] = torch.argmax(logits, dim=2)

        true_ids = b_labels.tolist()
        pred_ids = outputs["preds"].tolist()

        unpadded_true_labels = [[self.idx2tag.get(l) for l in sent if self.idx2tag.get(l)] for sent in true_ids]
        unpadded_pred_labels = [[self.idx2tag.get(l) for l in sent] for sent in pred_ids]

        for i in range(len(unpadded_true_labels)):
            unpadded_pred_labels[i] = unpadded_pred_labels[i][:len(unpadded_true_labels[i])]

        f1 = f1_score(unpadded_true_labels,unpadded_pred_labels,average='macro')
        acc = accuracy_score(unpadded_true_labels, unpadded_pred_labels)
        prec = precision_score(unpadded_true_labels, unpadded_pred_labels, average='macro')
        rec = recall_score(unpadded_true_labels, unpadded_pred_labels, average='macro')


        self.log("loss", outputs["loss"])
        self.log("f1_score", f1, on_epoch=True,logger=True, prog_bar=True)
        self.log("accuracy", acc, on_epoch=True, logger=True, prog_bar=True)
        self.log("precision", prec, on_epoch=True, logger=True, prog_bar=True)
        self.log("recall", rec, on_epoch=True, logger=True, prog_bar=True)




        return outputs

    def test_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        output = self.validation_step(batch, batch_idx)
        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        b_input_ids, b_input_mask, b_labels, b_docs = batch
        outputs = self.model(
            b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True
        )

        # compute the actual predicted classes for each token of each sequence in the batch
        logits = outputs["logits"]
        outputs["preds"] = torch.argmax(logits, dim=2)
        outputs["doc_ids"] = b_docs
        outputs["truths"] = b_labels
        outputs["tokens"] = b_input_ids

        return outputs

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        # return self.optimizer_cls(self.model.parameters(), lr=self.lr)
        return self.optimizer_cls(optimizer_grouped_parameters,     lr=3e-5,
                                  eps=1e-8)

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure):
    #     if self.trainer.global_step < 500:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams.learning_rate
    #
    #     optimizer.step()
    #     optimizer.zero_grad()

def __str__(self):
        return self.__class__.__name__
