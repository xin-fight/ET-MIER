# ET-MIER

# Requirements

Packages listed below are required.

- Python (tested on 3.7.2)
- CUDA (tested on 11.3)
- PyTorch (tested on 1.11.0)
- Transformers (tested on 4.14.1)
- pandas (tested on 1.3.5)
- scikit-learn (tested on 1.0.2)
- scipy (tested on 1.7.3)
- spacy (tested on 2.3.7)
- numpy (tested on 1.21.6)
- opt-einsum (tested on 3.3.0)
- ujson (tested on 5.1.0)
- tqdm (tested on 4.64.0)

> GPU: V100-32GB
>
> CPU: 6 vCPU Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz 

# Datasets

Our experiments include the [DocRED](https://github.com/thunlp/DocRED) and [Re-DocRED](https://github.com/tonytan48/Re-DocRED) datasets. The expected file structure is as follows:

**DocRED:**

```
ET-MIER
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- train_distant.json
 |-- meta
 |    |-- rel2id.json
 |-- feature_file
 |-- bert-base-cased
 |-- Logs
```

**ET-MIER-ReDocRED**

```
ET-MIER-ReDocRED
 |-- dataset
 |    |-- redocred
 |    |    |-- train_revised.json
 |    |    |-- train_annotated.json
 |    |    |-- dev_revised.json
 |    |    |-- test_revised.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- train_distant.json
 |-- meta
 |    |-- rel2id.json
 |-- feature_file
 |-- roberta-large
 |-- Redoc_Logs
```



# Training

## DocRED

### Model without evidence supervision

```shell
python run_iscf_SAIS_Evidence.py --do_train --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name bert_lambda0_seed66 --save_path bert_lambda0_seed66 --train_file train_annotated.json --dev_file dev.json --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 5e-5 --max_grad_norm 1.0 --evi_thresh 0.2 --evi_lambda 0 --warmup_ratio 0.06 --num_train_epochs 30.0 --num_class 97 --loss_weight_ET 0.25 --TSER_Logsumexp --three_atten
```

After training, the model weights are saved in `bert_lambda0_seed66/{model_dir}`

### Model with evidence supervision

```shell
python run_iscf_SAIS_Evidence.py --do_train --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --display_name bert_lambda0.1_seed66 --save_path bert_lambda0.1_seed66 --train_file train_annotated.json --dev_file dev.json --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 5e-5 --max_grad_norm 1.0 --evi_thresh 0.2 --evi_lambda 0.1 --warmup_ratio 0.06 --num_train_epochs 30.0 --num_class 97 --loss_weight_ET 0.25 --ishas_enhance_evi
```

After training, the model weights are saved in `bert_lambda0.1_seed66/{model_evi_dir}`

### The inference stage cross fusion strategy is applied

```shell
python run_iscf_SAIS_Evidence.py --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --load_path {model_evi_dir} --eval_mode single --test_file dev.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97 --ishas_enhance_evi
```

### Evaluation

```shell
python run_iscf_SAIS_Evidence.py --data_dir dataset/docred --transformer_type bert --model_name_or_path bert-base-cased --load_path {model_dir} --results_path {model_evi_dir} --eval_mode fushion --test_file dev.json --test_batch_size 32 --num_labels 4 --evi_thresh 0.2 --num_class 97 --loss_weight_ET 0.25 --TSER_Logsumexp --three_atten
```



## ReDocRED

### Model without evidence supervision

```shell
python run_iscf_SAIS_Evidence.py --do_train --data_dir dataset/redocred --transformer_type roberta --model_name_or_path roberta-large --train_file train_revised.json --dev_file dev_revised.json --save_path roberta_redocred_lambda0_seed66 --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 3e-5 --max_grad_norm 1.0 --evi_thresh 0.2 --evi_lambda 0 --warmup_ratio 0.06 --num_train_epochs 30.0 --num_class 97 --three_atten --Relation_loss_lambda 0.1 --loss_weight_ET 0.325 --Relation_loss_lambda 2 --s_attn_lambda 0.01
```

After training, the model weights are saved in `roberta_redocred_lambda0_seed66/{model_dir}`

### Model with evidence supervision

```shell
python run_iscf_SAIS_Evidence.py --do_train --data_dir dataset/redocred --transformer_type roberta --model_name_or_path roberta-large --train_file train_revised.json --dev_file dev_revised.json --save_path roberta_redocred_lambda0.1_seed66 --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 1 --num_labels 4 --lr_transformer 3e-5 --max_grad_norm 1.0 --evi_thresh 0.2 --evi_lambda 0.1 --warmup_ratio 0.06 --num_train_epochs 30.0 --num_class 97 --loss_weight_ET 0.25 --Relation_loss_lambda 2 --s_attn_lambda 0.01 --not_attn_loss --TSER_Logsumexp
```

After training, the model weights are saved in `roberta_redocred_lambda0.1_seed66/{model_evi_dir}`

### The inference stage cross fusion strategy is applied

```shell
python run_iscf_SAIS_Evidence.py --data_dir dataset/redocred --transformer_type roberta --model_name_or_path roberta-large --load_path {model_evi_dir} --eval_mode single --test_file dev_revised.json --test_batch_size 8 --num_labels 4 --evi_thresh 0.2 --num_class 97 --loss_weight_ET 0.25 --Relation_loss_lambda 2 --s_attn_lambda 0.01 --not_attn_loss --TSER_Logsumexp
```

### Evaluation

```shell
python run_iscf_SAIS_Evidence.py --data_dir dataset/redocred --transformer_type roberta --model_name_or_path roberta-large --load_path {model_dir} --results_path {model_evi_dir} --eval_mode fushion --test_file dev_revised.json --test_batch_size 32 --num_labels 4 --evi_thresh 0.2 --num_class 97 --three_atten --Relation_loss_lambda 0.1 --loss_weight_ET 0.325 --Relation_loss_lambda 2 --s_attn_lambda 0.01
```



# Additional Notes

The current structure of the code repository is relatively disorganized, and we plan to further reorganize it in the future (e.g., improving folder hierarchy and modularizing scripts) to make it easier for others to reproduce the results and maintain the code.
