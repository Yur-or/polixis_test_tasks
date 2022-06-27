#!/bin/bash

# pretrained model
export model_name='bert-base-uncased'

# path to dataset for training/finetune
export path_to_dataset='first_task/data/train.csv'

# output settings
export output_dir='first_task/outputs_bert/'

# misc. settings
export seed=19

# training settings
export num_train_epochs=4

# optimization settings
export learning_rate=3e-5
export epsilon=1e-8
export warmup_steps=0

# batch / sequence sizes
export batch_size=32
export max_seq_length=512

# percent division of the dataset
export train_prosentage=80
export valid_prosentage=10
export test_prosentage=10


python3 first_task/pipline_finetune_transformers/fine_tune_model.py \
    --model_name $model_name \
    --path_to_dataset $path_to_dataset \
    --output_dir $output_dir \
    --train_prosentage=$train_prosentage --valid_prosentage=$valid_prosentage --test_prosentage=$test_prosentage \
    --seed $seed \
    --num_train_epochs $num_train_epochs \
    --learning_rate=$learning_rate --epsilon=$epsilon --warmup_steps=$warmup_steps \
    --batch_size=$batch_size --max_seq_length=$max_seq_length
