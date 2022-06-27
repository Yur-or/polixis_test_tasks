# Polixis test tasks
Repository for solving a test task from Polixis

## Formulation of the problem:

### First task:
You need to train a topic classifier. The dataset can be found in folder data. <br />
There are no restrictions regarding algorithms/libraries/architectures to use. <br />
Please keep in mind that the code needs to be deployment-ready. <br />
Even though simpler solutions would also work we would like to see a transformer-based solution. <br />
You are also welcome to have more than one solution. <br />

### Second task:
For the second, easier task you are required to create a program that will return name equivalence pairs. <br />
You can find the dataset attached below. <br />
The dataset we provide contains a sample of English and Arabic equivalent names. <br />
Arabic names contain more than one equivalents, separated with simbold ، or ;  while English ones contain only one name. <br /> 
Your program should return Arabic-English pairs and Arabic-Arabic all possible pair combinations in separate files. <br />
Note that if ('a', 'c') and ('c', 'a') are the same pair and only one of them should be returned. <br />


## Tasks solutions:

Both solutions are written in Python 3.10.5, the necessary dependencies are described in requirements.txt

### First task:
I wrote a pipeline for fine-tune training of transformer models. I ran this fine-tune pipeline on 4 pretrained models. <br />
I will use the GPU: NVIDIA GeForce RTX 3090. <br />
Obtained results of metrics on test sampling:

1) distilbert-base-uncased (https://huggingface.co/distilbert-base-uncased) <br />
Time train: 18:15 <br />
Metrics: Accuracy: 0.952  |  F1 macro: 0.953  |  F1 micro: 0.952  |  MCC: 0.935

2) bert-base-uncased (https://huggingface.co/bert-base-uncased) <br />
Time train: 18:16 <br />
Metrics: Accuracy: 0.978  |  F1 macro: 0.978  |  F1 micro: 0.978  |  MCC: 0.97

3) sentence-transformers/sentence-t5-base (https://huggingface.co/sentence-transformers/sentence-t5-base) <br />
Time train: 18:16 <br />
Metrics: Accuracy: 0.957  |  F1 macro: 0.958  |  F1 micro: 0.957  |  MCC: 0.943

4) sentence-transformers/bert-base-nli-cls-token (https://huggingface.co/sentence-transformers/bert-base-nli-cls-token) <br />
Time train: 18:15 <br />
Metrics: Accuracy: 0.973  |  F1 macro: 0.974  |  F1 micro: 0.973  |  MCC: 0.964

This pipeline finds in the following way: first_task/pipline_finetune_transformers <br />
For run fine-tine you need to set the parameters in sh-script and the next run the following command in terminal on root of project: <br />
$ sh first_task/pipline_finetune_transformers/run.sh <br />

I also wrote API for inference models by FastApi package in the following way: first_task/API_inference <br />
To start server you need to run the following command while in this folder: <br />
$ uvicorn api:app --reload <br />
After that, you can use this API, the usage example is in file usage_api_server.py <br />

### Second task:
Path to solutin: second_task/csv_extractor.py <br />
There are 2 basic functions pairs_extractor_v1 and pairs_extractor_v2, which are 2 implementations of solving the task. <br />
The fisrt version is written with core python functionality, and the second version usesd the itertools package. <br />
The main function presents a call to these functions and a comparison of their operating time.
