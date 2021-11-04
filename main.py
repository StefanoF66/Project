from tqdm import trange

import torch

from transformers import RobertaTokenizer,RobertaForTokenClassification
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import f1_score, accuracy_score, classification_report
from seqeval.scheme import IOB2

from Dataloader import *
from Training import *
from preprocessing import *

MAX_WORD_LEN = 30
MAX_LEN = 45
bs = 32
MAX_CHUNKS_LEN = 40


model_name = "roberta-large"
csv_test = 'Data/buyer_target_seller 2/snp.buyer_target.test.csv'
csv_val = 'Data/buyer_target_seller 2/snp.buyer_target.dev.csv'
csv_train = 'Data/buyer_target_seller 2/snp.buyer_target.train.csv'

#Create 3 dataframes for test, val, and train

df_test, df_val, df_train = loading_data(csv_test,csv_val,csv_train)

#Create the dictionary and enum of tags

tags_list, tags_enum = dictionary_builder(df_test, df_val, df_train)

print(tags_list)
print(tags_enum)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(n_gpu)

# Initialize tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
sel_tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=False)

# prepare data for the transformer



# TRAIN

input_ids_train, tags_train, train_dataloader = preprocessing.preprocesser(df_train, sel_tokenizer, tags_enum, bs, MAX_LEN)

# VALIDATION

input_ids_val, tags_val, val_dataloader = preprocessing.preprocesser(df_val, sel_tokenizer, tags_enum, bs, MAX_LEN)



# TEST
tokenized_texts_test, labels_test, docs_test = preprocess_for_transformers(df_test, sel_tokenizer)

input_ids_test, tags_test, test_dataloader = preprocessing.preprocesser(df_test, sel_tokenizer, tags_enum, bs, MAX_LEN)

docs_test = [pad_list(doc_test, "[PAD]", max_l_size=MAX_LEN) for doc_test in docs_test]

#model settings

model = RobertaForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(tags_enum),
    output_attentions=False,
    output_hidden_states = False
)

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)


epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

model.to(device)




for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    val_preds, val_truths, val_loss = evaluate(model, val_dataloader)
    print("Validation loss: {}".format(val_loss))

    # Convert prediction and truth ids into their corresponding labels
    val_preds = map_ids2labels(val_preds, tags_list)
    val_truths = map_ids2labels(val_truths, tags_list)

    val_preds = remove_pad_from_list_of_sequences(val_preds, val_truths)  # remove predictions on pads
    val_truths = remove_pad_from_list_of_sequences(val_truths, val_truths)  # remove truths that are pads

    for i in range(20):
        print(f"Pred: {val_preds[i]}")
        print("----------------------------")
        print(f"Truth: {val_truths[i]}")
        print("==========================\n")

    print("Validation Accuracy: {}".format(accuracy_score(val_truths, val_preds)))
    print("Validation F1-Score: {}".format(f1_score(val_truths, val_preds)))
    print(classification_report(val_truths, val_preds, mode='strict', scheme=IOB2))

# ========================================
#               Test
# ========================================
# After the completion of each training epoch, measure our performance on
# our test set.
test_preds, test_truths, test_loss = evaluate(model, test_dataloader)
print("Test loss: {}".format(test_loss))

test_preds = map_ids2labels(test_preds, tags_list)
test_truths = map_ids2labels(test_truths, tags_list)

test_preds = remove_pad_from_list_of_sequences(test_preds, test_truths)  # remove predictions on pads
test_truths = remove_pad_from_list_of_sequences(test_truths, test_truths)  # remove truths that are pads

for i in range(20):
    print(f"Pred: {test_preds[i]}")
    print("----------------------------")
    print(f"Truth: {test_truths[i]}")
    print("==========================\n")

print("Test Accuracy: {}".format(accuracy_score(test_truths, test_preds)))
print("Test F1-Score: {}".format(f1_score(test_truths, test_preds)))
print(classification_report(test_truths, test_preds, mode='strict', scheme=IOB2))






