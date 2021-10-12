
import os
from copy import deepcopy
import json

from utils.commons import Extraction

from utils.basic_utils import is_loose_alias


class Occurrence:
    def __init__(self, start, end, value, entity_type, context=''):
        self.start = start
        self.end = end
        self.value = value
        self.entity_type = entity_type
        self.context = context

    def __repr__(self):
        return f"{self.entity_type} {self.value} - ({self.start},{self.end})"

    def __str__(self):
        return f"{self.entity_type} {self.value} - ({self.start},{self.end})"


class Opportunity:
    def __init__(self, doc_id, class_type, value):
        self.doc_id = doc_id
        self.class_type = class_type
        self.value = value
        self.occurrencies = []  # a list of Extraction objects

    def add_occurrency(self, occurrency):
        self.value = occurrency.value.replace("\\n", " ").replace("\n", " ") if len(occurrency.value) > len(self.value) else self.value
        self.occurrencies.append(occurrency)

    def n_occurrencies(self):
        return len(self.occurrencies)

    def is_alias(self, potential_alias, is_alias_func):
        for occurrence in self.occurrencies:
            if is_alias_func(occurrence, potential_alias):
                return True

        return False

    def unify(self, potential_aliases, is_alias_func):
        not_unified_aliases = []
        for alias in potential_aliases:
            if self.is_alias(alias, is_alias_func):
                self.add_occurrency(alias)
            else:
                not_unified_aliases.append(alias)

        return not_unified_aliases

    def to_dict(self):
        _dict = self.__dict__
        all_dict = deepcopy(_dict)
        all_dict["occurrencies"] = [occurrence.__dict__ for occurrence in _dict["occurrencies"]]

        return all_dict

    @staticmethod
    def group_opportunities(opportunities):
        opportunities_dict = {}
        for opp in opportunities:
            if opp.class_type not in opportunities_dict:
                opportunities_dict[opp.class_type] = []
            opportunities_dict[opp.class_type].append(opp)

        return opportunities_dict

    def already_exists(self, new_occ):
        for occ in self.occurrencies:
            if occ.start == new_occ.start:
                return True

        return False


class OpportunitiesMetrics:
    def __init__(self):
        pass

    def get_matches(self, true_per_class_opps, pred_per_class_opps, mode="strict"):
        matched_pairs = []
        matched_trues_ids, matched_preds_ids = [], []
        for i in range(len(true_per_class_opps)):
            if i not in matched_trues_ids:
                true_opp = true_per_class_opps[i]
                for j in range(len(pred_per_class_opps)):
                    if j not in matched_preds_ids:
                        pred_opp = pred_per_class_opps[j]
                        if self._opp_matches(true_opp, pred_opp, mode):
                            matched_pairs.append((true_opp, pred_opp))
                            matched_trues_ids.append(i)
                            matched_preds_ids.append(j)
                            break

        res = {
            "tps": matched_pairs,
            "fns": [opp for k, opp in enumerate(true_per_class_opps) if k not in matched_trues_ids],
            "fps": [opp for k, opp in enumerate(pred_per_class_opps) if k not in matched_preds_ids],
        }

        return res

    @staticmethod
    def _opp_matches(true_opp, pred_opp, mode="strict"):
        if mode == "strict":
            return pred_opp.value == true_opp.value
        else:
            return is_loose_alias(pred_opp.value, true_opp.value)

    def eval_opportunities(self, opportunities_path, opp_trues, opp_preds, prefix="test", mode="strict"):
        grouped_opps_true = [Opportunity.group_opportunities(opp) for opp in opp_trues]
        grouped_opps_preds = [Opportunity.group_opportunities(opp) for opp in opp_preds]

        opportunities_errors_json = os.path.join(opportunities_path, f"{prefix}_errors.json")
        self.log_opportunities_errors(grouped_opps_true, grouped_opps_preds, opportunities_errors_json, mode)

        metrics = self.compute_metrics(grouped_opps_true, grouped_opps_preds, mode)

        opportunities_metrics_json = os.path.join(opportunities_path, f"{prefix}_opportunities_report.json")
        with open(opportunities_metrics_json, 'w') as outfile:
            json.dump(metrics, outfile)

        return metrics

    def compute_metrics(self, docs_true_opportunities, docs_pred_opportunities, mode="strict"):
        tps, fps, fns = {}, {}, {}
        for o, true_opportunities in enumerate(docs_true_opportunities):
            pred_opportunities = docs_pred_opportunities[o]
            for k, true_opps in true_opportunities.items():
                    if k in pred_opportunities:
                        output = self.get_matches(true_opps, pred_opportunities[k], mode)
                        n_tps = len(output["tps"])
                        tps[k] = n_tps if k not in tps else tps[k] + n_tps
                        n_fps = len(output["fps"])
                        fps[k] = n_fps if k not in fps else fps[k] + n_fps
                        n_fns = len(output["fns"])
                        fns[k] = n_fns if k not in fns else fns[k] + n_fns
                    else:
                        tps[k] = 0 if k not in tps else tps[k] + 0
                        fps[k] = 0 if k not in fps else fps[k] + 0
                        fns[k] = len(true_opps) if k not in fns else fns[k] + len(true_opps)

        # adds FPs to documents with no ground truth opportunities of kind k
        for o, pred_opportunities in enumerate(docs_pred_opportunities):
            for k, pred_opps in pred_opportunities.items():
                if docs_true_opportunities[o] and k in docs_true_opportunities[o]:
                    continue
                else:
                    fps[k] = len(pred_opps) if k not in fps else fps[k] + len(pred_opps)

        opportunities_metrics = {}
        for k in fns.keys():
            opportunities_metrics[k] = {}
            tp_fn = tps[k] + fns[k]  # TP + FN, i.e all the trues
            tp_fp = tps[k] + fps[k]

            opp_recall = tps[k]/tp_fn if tp_fn > 0 else 0  # TP/(TP+FN)
            opp_precision = tps[k]/tp_fp if tp_fp > 0 else 0  # TP/(TP+FP)

            opp_f1 = 2*(opp_recall * opp_precision)/(opp_recall+opp_precision) if opp_recall+opp_precision > 0 else 0  # TP/(TP+FP)

            imprecision_rate = fps[k]/tp_fp if tp_fp > 0 else 0  # FP/ (FP + TP) todo check this i am not sure it is the proper score

            support = tp_fn

            opportunities_metrics[k] = {
                "precision": opp_precision,
                "recall": opp_recall,
                "f1-score": opp_f1,
                "imprecision_rate": imprecision_rate,
                "support": support
            }

        return opportunities_metrics

    def log_opportunities_errors(self, docs_true_opportunities, docs_pred_opportunities, json_file, mode="strict"):
        errors = {"FPs": [], "FNs": [], "TPs": []}
        for o, true_opportunities in enumerate(docs_true_opportunities):
            pred_opportunities = docs_pred_opportunities[o]

            for k, true_opps in true_opportunities.items():
                pred_opps_k = pred_opportunities[k] if k in pred_opportunities else []
                output = self.get_matches(true_opps, pred_opps_k, mode)
                errors["FPs"].extend([fp.to_dict() for fp in output["fps"]])
                errors["FNs"].extend([fn.to_dict() for fn in output["fns"]])
                errors["TPs"].extend([{"true": tp_pair[0].to_dict(), "pred": tp_pair[1].to_dict()} for tp_pair in output["tps"]])

        # adds FPs to documents with no ground truth opportunities of kind k
        for o, pred_opportunities in enumerate(docs_pred_opportunities):
            for k, pred_opps in pred_opportunities.items():
                if docs_true_opportunities[o] and k in docs_true_opportunities[o]:
                    continue
                else:
                    errors["FPs"].extend([fp.to_dict() for fp in pred_opps])

        with open(json_file, 'w') as outfile:
            json.dump(errors, outfile)


class AnnotationOpportunity(Opportunity):
    def __init__(self, doc_id, class_type, value):
        super(AnnotationOpportunity, self).__init__(doc_id, class_type, value)

    def add_occurrency(self, ann_occurrence):
        ann_occurrence = Occurrence(
            ann_occurrence.positions[0].start,
            ann_occurrence.positions[0].end,
            ann_occurrence.normalized_value.replace("\\n", " ").replace("\n", " "),
            ann_occurrence.label
        )
        super(AnnotationOpportunity, self).add_occurrency(ann_occurrence)

    @staticmethod
    def get_opportunities_from_annotations(doc):
        def is_alias(occurrence, alias):
            return occurrence.entity_type == alias.label and is_loose_alias(occurrence.value, alias.normalized_value)

        return AnnotationOpportunity._get_opportunities(doc.annotations.elements, is_alias, doc.document_id)

    @staticmethod
    def _get_opportunities(elements, is_alias_func, doc_id):
        unmerged_annotations = deepcopy(elements)
        opportunities = []
        while len(unmerged_annotations) > 0:
            unique_entity = unmerged_annotations[0]
            opportunities.append(AnnotationOpportunity(doc_id, unique_entity.label, unique_entity.normalized_value.strip().replace("\\n", " ").replace("\n", " ")))
            opportunities[-1].add_occurrency(unique_entity)
            unmerged_annotations = opportunities[-1].unify(unmerged_annotations[1:], is_alias_func)

        return opportunities


class ExtractionOpportunity(Opportunity):
    def __init__(self, doc_id, class_type, value):
        super(ExtractionOpportunity, self).__init__(doc_id, class_type, value)

    def add_occurrency(self, extr_occurrency): # fixme name
        entity_type = f"{extr_occurrency.template}.{extr_occurrency.fields[0].name}"
        content_range = 25
        for pos in extr_occurrency.fields[0].positions:
            start = pos.start
            end = pos.end
            val = extr_occurrency.fields[0].val
            occ = Occurrence(start, end, val, entity_type)
            super(ExtractionOpportunity, self).add_occurrency(occ)

    def add_occurrence(self, occ):
        super(ExtractionOpportunity, self).add_occurrency(occ)

    @staticmethod
    def get_opportunities_from_extractions(doc):
        def is_alias(occurrence, alias):
            return occurrence.entity_type == f"{alias.template}.{alias.fields[0].name}" and is_loose_alias(occurrence.value, alias.fields[0].val)

        return ExtractionOpportunity._get_opportunities(doc.extractions.elements, is_alias, doc.document_id)

    @staticmethod
    def get_opportunities_from_transformers(document_id, xs, y, tokenizer):
        def is_alias(occurrence, alias):
            return occurrence.entity_type == f"{alias.template}.{alias.fields[0].name}" and is_loose_alias(occurrence.value, alias.fields[0].val)

        return ExtractionOpportunity._get_opportunities(IOB2ml_extractions(xs, y, tokenizer), is_alias, document_id)

    @staticmethod
    def _get_opportunities(elements, is_alias_func, doc_id):
        unmerged_extractions = deepcopy(elements)
        unmerged_extractions = ExtractionOpportunity.linearize_extractions(unmerged_extractions)  # creates one-field only extractions
        opportunities = []
        while len(unmerged_extractions) > 0:
            unique_entity = unmerged_extractions[0]
            opportunities.append(ExtractionOpportunity(doc_id, f"{unique_entity.template}.{unique_entity.fields[0].name}", unique_entity.fields[0].val.strip()))
            opportunities[-1].add_occurrency(unique_entity)
            unmerged_extractions = opportunities[-1].unify(unmerged_extractions[1:], is_alias_func)

        return opportunities

    @staticmethod
    def linearize_extractions(extractions):
        linearized_extractions = []
        for extraction in extractions:
            for field in extraction.fields:
                extr = Extraction(document=extraction.document, namespace=extraction.namespace, template=extraction.template)
                extr.add_field(name=field.name,
                               val=field.val,
                               positions_dicts=[{"start": pos.start, "end": pos.end} for pos in field.positions])
                linearized_extractions.append(extr)

        return linearized_extractions


def IOB2ml_extractions(xs, iob_ys, tokenizer, namespace=''):
    """
    Converts IOB predictions into ml_extractions
    :param xs: list of tokens
    :param iob_ys: list of labels in IOB format (list of lists of strings)
    :return: a list of extractions
    """
    extractions = []
    extraction_tokens = []
    cls_type = None
    for j, token in enumerate(xs):
        if iob_ys[j] == "O":
            if extraction_tokens and cls_type:
                template, field = cls_type.split(".")
                extr = Extraction(document=None, namespace=namespace, template=template)
                extr.add_field(name=field,
                               val=tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(extraction_tokens)),
                               positions_dicts=[{
                                   "start": -1,
                                   "end": -1
                               }])

                extractions.append(extr)

            extraction_tokens = []
        else:
            cls_type = iob_ys[j].replace("B-", "").replace("I-", "")
            extraction_tokens.append(token)

    return extractions
