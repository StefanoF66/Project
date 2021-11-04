import torch

from pytorch_lightning import Trainer

from utils.opportunities import ExtractionOpportunity, OpportunitiesMetrics


def linearize(seqs):
    return [el for seq in seqs for el in seq]


class NerTrainer(Trainer):
    def __init__(self, **kwargs):
        super(NerTrainer, self).__init__(**kwargs)

    def predict(self, model=None, dataloaders=None, datamodule=None, return_predictions=None, ckpt_path="best"):
        outputs = super(NerTrainer, self).predict(model, dataloaders, datamodule, return_predictions, ckpt_path)
        truths = torch.cat([out["truths"] for out in outputs], dim=0).detach().cpu().numpy()
        truths_mask = truths >= 0
        # todo clean up
        preds = torch.cat([out["preds"] for out in outputs], dim=0).detach().cpu().numpy()
        preds = preds[truths_mask]
        # preds = [[self.datamodule.idx2tag[tok] for tok in seq if tok in self.datamodule.idx2tag] for seq in preds]
        preds = [self.datamodule.idx2tag[tok] for tok in preds if tok in self.datamodule.idx2tag]
        linear_preds = preds
        # print(preds)

        truths = torch.cat([out["truths"] for out in outputs], dim=0).detach().cpu().numpy()
        truths = [[self.datamodule.idx2tag[tok] for tok in seq if tok in self.datamodule.idx2tag] for seq in truths]
        linear_truths = linearize(truths)
        # print(truths)

        docs = torch.cat([out["doc_ids"] for out in outputs], dim=0).detach().cpu().numpy()
        docs = docs[truths_mask]
        docs = [self.datamodule.idx2docid[tok] for tok in docs if tok in self.datamodule.idx2docid]
        # docs = [[self.datamodule.idx2docid[tok] for tok in seq if tok in self.datamodule.idx2docid] for seq in docs]
        linear_docs = docs
        # linear_docs = linearize(docs)
        unique_docs = set(linear_docs)
        # print(docs)

        tokens = torch.cat([out["tokens"] for out in outputs], dim=0).detach().cpu().numpy()
        tokens = tokens[truths_mask]
        tokens = [tok for tok in tokens if self.datamodule.tokenizer.convert_ids_to_tokens(int(tok)) != "[PAD]"]
        linear_tokens = tokens
        # print(tokens)

        outputs = list(zip(linear_tokens, linear_truths, linear_preds, linear_docs))
        # print(outputs)

        def get_tuple_of_doc(output_tuple, doc_id):
            return [t for t in output_tuple if t[-1] == doc_id]

        truth_opps, pred_opps = [], []
        for doc_id in unique_docs:
            doc = get_tuple_of_doc(outputs, doc_id)
            xs, trues, predictions, _ = list(zip(*doc))
            # print(xs)
            truth_opps.append(ExtractionOpportunity.get_opportunities_from_transformers(doc_id, xs, trues, self.datamodule.tokenizer))
            pred_opps.append(ExtractionOpportunity.get_opportunities_from_transformers(doc_id, xs, predictions, self.datamodule.tokenizer))

        # print([[opp.value for opp in doc_opps] for doc_opps in truth_opps])
        # print([[opp.value for opp in doc_opps] for doc_opps in pred_opps])
        opp_metrics = OpportunitiesMetrics()
        results = opp_metrics.eval_opportunities('', truth_opps, pred_opps, prefix="tolerant", mode="tolerant")
        results = opp_metrics.eval_opportunities('', truth_opps, pred_opps, prefix="strict", mode="strict")
        # print(results)

        return preds
