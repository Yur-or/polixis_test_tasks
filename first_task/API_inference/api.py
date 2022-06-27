from typing import List

from transformers import BertForSequenceClassification, AutoTokenizer
from fastapi import FastAPI, HTTPException, Query
import torch


tokenizer_distilbert = AutoTokenizer.from_pretrained('../outputs_distilbert', use_auth_token=True)
model_distilbert = BertForSequenceClassification.from_pretrained('../outputs_distilbert', use_auth_token=True)

tokenizer_bert = AutoTokenizer.from_pretrained('../outputs_bert', use_auth_token=True)
model_bert = BertForSequenceClassification.from_pretrained('../outputs_bert', use_auth_token=True)

tokenizer_sbert = AutoTokenizer.from_pretrained('../outputs_sbert', use_auth_token=True)
model_sbert = BertForSequenceClassification.from_pretrained('../outputs_sbert', use_auth_token=True)

tokenizer_t5 = AutoTokenizer.from_pretrained('../outputs_t5', use_auth_token=True)
model_t5 = BertForSequenceClassification.from_pretrained('../outputs_t5', use_auth_token=True)


app = FastAPI(
    title="API for classification",
    description="A simple API that inference fine-tune transformers model for classification",
    version="1",
)


def get_predict_batch(model: BertForSequenceClassification, 
    tokenizer: AutoTokenizer, 
    texts: List[str]) -> List[dict]:

    model.eval()

    encoded_dict = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        result = model(encoded_dict['input_ids'],
                        token_type_ids=None,
                        attention_mask=encoded_dict['attention_mask'],
                        return_dict=True)

    func_softmax = torch.nn.Softmax(dim=1)

    res_probabilite = func_softmax(result.logits)

    return res_probabilite.tolist()


def get_predict(model: BertForSequenceClassification, 
    tokenizer: AutoTokenizer, 
    text: str) -> dict:

    model.eval()

    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        result = model(encoded_dict['input_ids'],
                        token_type_ids=None,
                        attention_mask=encoded_dict['attention_mask'],
                        return_dict=True)

    func_softmax = torch.nn.Softmax(dim=1)

    return func_softmax(result.logits).tolist()


@app.get("/predict")
def predict_label(model_name: str, text: str):

    match model_name:
        case 'distilbert':
            return get_predict(model_distilbert, tokenizer_distilbert, text)
        case 'bert':
            return get_predict(model_bert, tokenizer_bert, text)
        case 'sbert':
            return get_predict(model_sbert, tokenizer_sbert, text)
        case 't5':
            return get_predict(model_t5, tokenizer_t5, text)
        case _:
            raise HTTPException(status_code=404, detail='Choose one of the models (distilbert, bert, sbert, t5)')


@app.get("/predict_batch")
def predict_label_batch(model_name: str, texts: List[str] = Query(None)):

    match model_name:
        case 'distilbert':
            return get_predict_batch(model_distilbert, tokenizer_distilbert, texts)
        case 'bert':
            return get_predict_batch(model_bert, tokenizer_bert, texts)
        case 'sbert':
            return get_predict_batch(model_sbert, tokenizer_sbert, texts)
        case 't5':
            return get_predict_batch(model_t5, tokenizer_t5, texts)
        case _:
            raise HTTPException(status_code=404, detail='Choose one of the models (distilbert, bert, sbert, t5)')
