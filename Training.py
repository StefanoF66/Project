import numpy as np
from tqdm import trange
import torch
from transformers import RobertaTokenizer,RobertaForTokenClassification
from transformers import AdamW
from keras.preprocessing.sequence import pad_sequences
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, accuracy_score, classification_report
from seqeval.scheme import IOB2

from utils.preprocessing import preprocess_for_transformers, make_dataloader, pad_list


def id2label(seq, tag_values):
    return [tag_values[el] for el in seq]


def map_ids2labels(seqs, tag_values):
    return [id2label(seq, tag_values) for seq in seqs]


def remove_pad(sequence, truth_seq, pad_tag="PAD"):
    return [el for i,el in enumerate(sequence) if truth_seq[i] != pad_tag]


def remove_pad_from_list_of_sequences(seqs, truth_seqs, pad_tag="PAD"):
    cleaned_seq = [remove_pad(seq, truth_seq, pad_tag) for seq, truth_seq in zip(seqs, truth_seqs)]
    return [[el if el != pad_tag else "O" for el in seq] for seq in cleaned_seq]


def evaluate(model, samples_dataloader, device="cuda"):
    model.eval()
    eval_loss = 0
    preds, truths = [], []
    for batch in samples_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(input_ids, token_type_ids=None,
                            attention_mask=input_mask, labels=labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        labels = labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        preds.extend([list(p) for p in np.argmax(logits, axis=2)])
        truths.extend(labels)
        eval_loss += outputs[0].mean().item()

    return preds, truths, eval_loss