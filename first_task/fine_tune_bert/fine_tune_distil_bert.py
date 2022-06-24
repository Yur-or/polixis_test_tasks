from dataclasses import dataclass
from typing import Tuple
import argparse
import datetime
import logging
import random
import time
import os

from transformers import BertForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from dacite import from_dict
import pandas as pd
import numpy as np
import torch
import yaml
import wandb


def init_logger() -> None:
    global log

    with open('first_task/fine_tune_bert/logging_conf.yaml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        logging.config.dictConfig(config)
        log = logging.getLogger('Main')


@dataclass
class TrainArguments:
    model_name: str = None
    path_to_dataset: str = None
    output_dir: str = None
    train_prosentage: int = None
    valid_prosentage: int = None
    test_prosentage: int = None
    seed: int = None
    num_train_epochs: int = None
    learning_rate: float = None
    epsilon: float = None
    warmup_steps: int = None
    batch_size: int = None
    max_seq_length: int = None


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description='Argumetns for training')

    parser.add_argument('--model_name', default=None, required=True, type=str)
    parser.add_argument('--path_to_dataset', default=None, required=True, type=str)
    parser.add_argument('--output_dir', default=None, required=True, type=str)
    parser.add_argument('--train_prosentage', default=None, required=True, type=int)
    parser.add_argument('--valid_prosentage', default=None, required=True, type=int)
    parser.add_argument('--test_prosentage', default=None, required=True, type=int)
    parser.add_argument('--seed', default=None, required=True, type=int)
    parser.add_argument('--num_train_epochs', default=None, required=True, type=int)
    parser.add_argument('--learning_rate', default=None, required=True, type=float)
    parser.add_argument('--epsilon', default=None, required=True, type=float)
    parser.add_argument('--warmup_steps', default=None, required=True, type=int)
    parser.add_argument('--batch_size', default=None, required=True, type=int)
    parser.add_argument('--max_seq_length', default=None, required=True, type=int)

    args = from_dict(data_class=TrainArguments, data=vars(parser.parse_args()))

    log.info(f'Train arguments: {args}')

    return args


def init_wandb(lr: float, count_epochs) -> None:
    wandb.init(
        project='polixis_test_task', 
        name='fine_tune_distil_bert',
        config={
        'learning_rate': lr,
        'architecture': 'BertForSequenceClassification',
        'epochs': count_epochs
        }
    )


def check_gpu_and_init_device() -> None:
    global device

    if torch.cuda.is_available():    

        device = torch.device('cuda')
        log.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        log.info(f'We will use the GPU: {torch.cuda.get_device_name(0)}')

    else:
        log.warning('No GPU available, using the CPU instead.')
        device = torch.device('cpu')


def format_time(secons: float) -> str:
    return str(datetime.timedelta(seconds=secons))


def flat_accuracy(preds: np.array, labels: np.array) -> float:
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_and_preprocess_data(path_to_data: os.PathLike) -> pd.DataFrame:

    df = pd.read_csv(path_to_data)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset='text', keep=False, inplace=True)

    return df


def tokenize_and_create_TensorDataset(texts: pd.Series, labels: pd.Series, tokenizer: AutoTokenizer) -> TensorDataset:
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,  
                            add_special_tokens = True,
                            max_length = 512,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                    )
        
        input_ids.append(encoded_dict['input_ids'])
        
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.values)

    return TensorDataset(input_ids, attention_masks, labels)


@dataclass
class SplitPercentage:
    train_percentage: int = 0
    valid_percentage: int = 0
    test_percentage: int = 0

    def __post_init__(self):
        if sum((self.train_percentage, self.test_percentage, self.valid_percentage)) != 100:
            raise Exception('Sum percentages must be equal 100')


def split_data_and_create_DataLoaders(dataset: TensorDataset, split_prosentage: SplitPercentage, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    fist_spliter = StratifiedShuffleSplit(n_splits=1, train_size=split_prosentage.train_percentage/100, random_state=seed)
    train_indexes, val_test_indexes = next(fist_spliter.split(dataset, dataset[:][2]))
    train_dataset = Subset(dataset, train_indexes)
    val_test_dataset = Subset(dataset, val_test_indexes)
    
    second_spliter = StratifiedShuffleSplit(n_splits=1, train_size=split_prosentage.valid_percentage/(split_prosentage.valid_percentage + split_prosentage.test_percentage), random_state=seed)
    val_indexes, test_indexes = next(second_spliter.split(val_test_dataset, val_test_dataset[:][2]))
    val_dataset = Subset(val_test_dataset, val_indexes)
    test_dataset = Subset(val_test_dataset, test_indexes)

    train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )

    valid_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )
    
    test_dataloader = DataLoader(
            test_dataset,
            sampler = SequentialSampler(test_dataset),
            batch_size = batch_size
        )
    
    return train_dataloader, valid_dataloader, test_dataloader


def train(model: BertForSequenceClassification, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, scheduler: torch.optim.lr_scheduler.LambdaLR, optimizer: AdamW, seed: int) -> BertForSequenceClassification:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    total_t0 = time.perf_counter()

    for epoch_i in range(1, epochs + 1):
        
        log.info(f'======== Epoch {epoch_i} / {epochs} ========')
        log.info('Training...')

        t0 = time.perf_counter()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.perf_counter() - t0)
                log.info(f' Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}.')
                log.info(f'Current learning rate {scheduler.get_lr()}')
                wandb.log({'lr': scheduler.get_lr()})

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels,
                        return_dict=True)

            loss = result.loss

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            log.info(f'Train loss: {loss:.3f}')
            wandb.log({'train_loss': loss})

        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        training_time = format_time(time.perf_counter() - t0)

        log.info(f' Average training loss: {avg_train_loss:.3f}')
        log.info(f' Training epcoh took: {training_time}')
            
        #Validation

        log.info('Running Validation...')

        t0 = time.perf_counter()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in valid_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        

                result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True)

            loss = result.loss
                
            total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(valid_dataloader)
        
        validation_time = format_time(time.perf_counter() - t0)
        
        log.info(f' Validation Loss: {avg_val_loss:.3f}')
        log.info(f' Validation took: {validation_time}')

        wandb.log({
            'epoch': epoch_i,
            'time': training_time + validation_time,
            'avg_train_loss': avg_train_loss,
            'avg_valid_loss': avg_val_loss
        })

    log.info('Training complete!')
    log.info(f'Total training took {format_time(time.perf_counter()-total_t0)} (h:mm:ss)')

    return model


def evalute_test(model: BertForSequenceClassification, prediction_dataloader: DataLoader) -> None:

    log.info(f'Predicting labels for {len(prediction_dataloader.dataset)} test sentences...')

    model.eval()

    predictions, true_labels = [], []

    for batch in prediction_dataloader:
        
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            return_dict=True)

        logits = result.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.append(logits)
        true_labels.append(label_ids)

    metrics = calculation_metrics(predictions, true_labels)

    log.info('Relust test metrics:')
    log.info(f'Accuracy: {metrics.accuracy:.3f}')
    log.info(f'F1 macro: {metrics.f1_macro:.3f}')
    log.info(f'F1 micro: {metrics.f1_micro:.3f}')
    log.info(f'MCC: {metrics.mcc:.3f}')

    wandb.log({
        'accuracy': metrics.accuracy,
        'f1_macro': metrics.f1_macro,
        'f1_micro': metrics.f1_micro,
        'mcc': metrics.mcc
    })

    log.info('DONE!')


@dataclass
class SetMetrics:
    accuracy: float = None
    f1_macro: float = None
    f1_micro: float = None
    mcc: float = None


def calculation_metrics(predictions: np.array, true_labels: np.array) -> SetMetrics:

    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)

    metrics = SetMetrics(
        accuracy = accuracy_score(flat_true_labels, flat_predictions),
        f1_macro = f1_score(flat_true_labels, flat_predictions, average='macro'),
        f1_micro = f1_score(flat_true_labels, flat_predictions, average='micro'),
        mcc = matthews_corrcoef(flat_true_labels, flat_predictions),
    )

    return  metrics


def save_trained_model(model: BertForSequenceClassification, output_dir: os.PathLike, tokenizer: AutoTokenizer) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log.info(f'Saving model to {output_dir}')

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    
    init_logger()
    args = parse_arguments()
    init_wandb(args.learning_rate, args.num_train_epochs)
    check_gpu_and_init_device()

    df = get_and_preprocess_data(args.path_to_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = tokenize_and_create_TensorDataset(df['text'], df['cls'], tokenizer)

    split_prosentage = SplitPercentage(args.train_prosentage, args.valid_prosentage, args.test_prosentage)

    train_dataloader, valid_dataloader, test_dataloader = split_data_and_create_DataLoaders(dataset, split_prosentage, args.batch_size, args.seed)

    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels = 4,
        output_attentions = False,
        output_hidden_states = False,
    )

    model.to(device)

    optimizer = AdamW(model.parameters(),
                  lr = args.learning_rate,
                  eps = args.epsilon
    )

    total_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = args.warmup_steps,
                                                num_training_steps = total_steps
    )

    trained_model = train(model, train_dataloader, valid_dataloader, args.num_train_epochs, scheduler, optimizer, args.seed)

    evalute_test(trained_model, test_dataloader)

    save_trained_model(trained_model, args.output_dir, tokenizer)

    wandb.finish()


if __name__ == '__main__':
    main()
