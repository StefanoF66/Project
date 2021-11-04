"""
Script to run experiments for NER with transformers with pytorch-lightning

Lightining main principles:
    1. Disentangle development of data, models and training
    2. DataModules (e.g. TransformerFromConllDFDataModule) load the data, process them and
    provides dataloaders
    3. LightingModules are useful to wrap the transformer models beside the optimization stuff

"""

from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import RobertaForTokenClassification
from transformers import BertForTokenClassification
from trainers import NerTrainer

from datasets import TransformerFromConllDFDataModule
from models import TransformerModule


if __name__ == "__main__":

    df_train_path = 'data/snp.buyer_target.train.csv'
    df_val_path = 'data/snp.buyer_target.dev.csv'
    df_test_path = 'data/snp.buyer_target.test.csv'

    model_name = "bert-large"
    if "roberta" in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=False)
    elif "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    else:
        raise NotImplementedError

    data_module = TransformerFromConllDFDataModule(
        df_train_path, df_val_path, df_test_path, tokenizer, train_batch_size=64, eval_batch_size=128, sample_perc=1.0, max_token_seq_len=104, max_word_seq_len=80
    )

    if "roberta" in model_name:
        model = RobertaForTokenClassification.from_pretrained(
            model_name,
            num_labels=data_module.n_classes,
            output_attentions=False,
            output_hidden_states=False
        )
    elif "bert" in model_name:
        model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=data_module.n_classes,
            output_attentions=False,
            output_hidden_states=False
        )
    else:
        raise NotImplementedError

    model_module = TransformerModule(lr=1e-3, model=model, idx2tag=data_module.idx2tag, tokenizer=tokenizer)

    n_epochs = 3
    trainer = NerTrainer(gpus=None, max_epochs=n_epochs, gradient_clip_val=2.0, log_every_n_steps=200)
    trainer.fit(model=model_module, datamodule=data_module)
    trainer.predict(dataloaders=[data_module.test_dataloader()])