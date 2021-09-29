from src.data.universal_dataset import UniversalDataset, uni_labels, get_transform_labels_from_batch_labels
from src.config import Config
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, PreTrainedTokenizer
from tqdm import tqdm
import argparse
from src.utils import get_optimizers, write_data
import torch
import torch.nn as nn
import numpy as np
import os
import random
from src.model.parallel_model import ParallelModel
from collections import Counter
from src.eval.utils import is_value_correct
from typing import List
from torch.nn.utils.rnn import pad_sequence

# torch.autograd.set_detect_anomaly(True)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("[Info] YOU ARE USING CPU, change --device to cuda if you are using GPU")

def parse_arguments(parser:argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], help="GPU/CPU devices")
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--shuffle_train_data', type=int, default=1, choices=[0, 1], help="shuffle the training data or not")
    parser.add_argument('--train_num', type=int, default=5, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=5, help="The number of development data, -1 means all data")


    parser.add_argument('--train_file', type=str, default="data/math23k/train23k_parallel_sorted.json")
    parser.add_argument('--dev_file', type=str, default="data/math23k/test23k_parallel_sorted.json")

    parser.add_argument('--filtered_steps', default=None, nargs='+', help="some heights to filter")
    parser.add_argument('--use_constant', default=1, type=int, choices=[0,1], help="whether to use constant 1 and pi")

    parser.add_argument('--add_replacement', default=1, type=int, choices=[0,1], help = "use replacement when computing combinations")

    parser.add_argument('--add_new_token', default=0, type=int, choices=[0, 1], help="whether or not to add the new token")

    # model
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--model_folder', type=str, default="math_solver", help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="hfl", help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="chinese-roberta-wwm-ext",
                        help="The bert model name to used")
    parser.add_argument('--diff_param_for_height', type=int, default=0, choices=[0,1])
    parser.add_argument('--height', type=int, default=10, help="the model height")
    parser.add_argument('--consider_multiple_m0', type=int, default=1, help="whether or not to consider multiple m0")

    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="learning rate of the AdamW optimizer")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate of the AdamW optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")
    parser.add_argument('--num_epochs', type=int, default=20, help="The number of epochs to run")
    parser.add_argument('--temperature', type=float, default=1.0, help="The temperature during the training")
    parser.add_argument('--fp16', type=int, default=0, choices=[0,1], help="using fp16 to train the model")

    parser.add_argument('--parallel', type=int, default=0, choices=[0,1], help="parallelizing model")

    # testing a pretrained model
    parser.add_argument('--cut_off', type=float, default=-100, help="cut off probability that we don't want to answer")
    parser.add_argument('--print_error', type=int, default=0, choices=[0, 1], help="whether to print the errors")
    parser.add_argument('--error_file', type=str, default="results/error.json", help="The file to print the errors")
    parser.add_argument('--result_file', type=str, default="results/res.json",
                        help="The file to print the errors")

    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train(config: Config, train_dataloader: DataLoader, num_epochs: int,
          bert_model_name: str, num_labels: int,
          dev: torch.device, tokenizer: PreTrainedTokenizer, valid_dataloader: DataLoader = None,
          constant_values: List = None):

    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * num_epochs)

    constant_num = len(constant_values) if constant_values else 0
    model = ParallelModel.from_pretrained(bert_model_name,
                                           diff_param_for_height=config.diff_param_for_height,
                                           num_labels=num_labels,
                                           height=config.height,
                                           constant_num=constant_num).to(dev)
    if config.add_new_token:
        model.resize_token_embeddings(len(tokenizer))
        if model.config.tie_word_embeddings:
            model.tie_weights()
        print(f"[Info] Added new tokens <NUM> for grading purpose, after: {len(tokenizer)}")


    if config.parallel:
        model = nn.DataParallel(model)

    scaler = None
    if config.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config.fp16))

    optimizer, scheduler = get_optimizers(config, model, t_total)
    model.zero_grad()

    best_performance = 10000000
    os.makedirs(f"model_files/{config.model_folder}", exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for iter, feature in tqdm(enumerate(train_dataloader, 1), desc="--training batch", total=len(train_dataloader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                loss = model(input_ids=feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                             token_type_ids=feature.token_type_ids.to(dev),
                             variable_indexs_start=feature.variable_indexs_start.to(dev),
                             variable_indexs_end=feature.variable_indexs_end.to(dev),
                             num_variables = feature.num_variables.to(dev),
                             variable_index_mask= feature.variable_index_mask.to(dev),
                             labels=feature.labels.to(dev),
                             return_dict=True).loss
            if config.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            total_loss += loss.item()
            if config.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            model.zero_grad()
            if iter % 1000 == 0:
                print(f"epoch: {epoch}, iteration: {iter}, current mean loss: {total_loss/iter:.2f}", flush=True)
        print(f"Finish epoch: {epoch}, loss: {total_loss:.2f}, mean loss: {total_loss/len(train_dataloader):.2f}", flush=True)
        if valid_dataloader is not None:
            # performance = evaluate(valid_dataloader, model, dev, fp16=bool(config.fp16), constant_values=constant_values)
            performance = evaluate_loss(valid_dataloader, model, dev, fp16=bool(config.fp16))
            if performance < best_performance:
                print(f"[Model Info] Saving the best model... with performance {performance}..")
                best_performance = performance
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(f"model_files/{config.model_folder}")
                tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    print(f"[Model Info] Best validation performance: {best_performance}")
    model = ParallelModel.from_pretrained(f"model_files/{config.model_folder}",
                                           diff_param_for_height=config.diff_param_for_height,
                                           num_labels=num_labels,
                                           height=config.height,
                                           constant_num=constant_num).to(dev)
    if config.fp16:
        model.half()
        model.save_pretrained(f"model_files/{config.model_folder}")
        tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    return model


def check_multiple_stops_and_remove_logit(batched_prediction):
    for b, inst_predictions in enumerate(batched_prediction):
        ## filter out empy first
        for p, prediction_steps in enumerate(inst_predictions):
            if len(prediction_steps) == 0:
                batched_prediction[b] = batched_prediction[b][:p]

        if len(batched_prediction[b]) > 0:
            last_equations = batched_prediction[b][-1]
            if len(last_equations) > 1:
                stop_equation_idxs = []
                for idx, last_equation in enumerate(last_equations):
                    left, right, op_label, stop_id = last_equation
                    if stop_id == 1:
                        stop_equation_idxs.append(idx)
                if len(stop_equation_idxs) > 0:
                    batched_prediction[b][-1] = [last_equations[stop_equation_idxs[0]]]
                else:
                    batched_prediction[b][-1] = [last_equations[0]]
    return batched_prediction

def evaluate_loss(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device, fp16:bool) -> float:
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                valid_loss = model(input_ids=feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                                   token_type_ids=feature.token_type_ids.to(dev),
                                   variable_indexs_start=feature.variable_indexs_start.to(dev),
                                   variable_indexs_end=feature.variable_indexs_end.to(dev),
                                   num_variables=feature.num_variables.to(dev),
                                   variable_index_mask=feature.variable_index_mask.to(dev),
                                   labels=feature.labels.to(dev),
                                   return_dict=True,
                                   is_eval=False).loss  ##List of (batch_size, num_combinations/num_m0, num_labels, 2, 2)
                total_valid_loss += valid_loss.item()
    print(f"Total validation loss: {total_valid_loss}")
    return total_valid_loss

def evaluate(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device, fp16:bool, constant_values: List, res_file: str= None, err_file:str = None) -> float:
    model.eval()
    predictions = []
    labels = []
    constant_num = len(constant_values) if constant_values else 0
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                all_logits = model(input_ids= feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                             token_type_ids= feature.token_type_ids.to(dev),
                             variable_indexs_start= feature.variable_indexs_start.to(dev),
                             variable_indexs_end= feature.variable_indexs_end.to(dev),
                             num_variables = feature.num_variables.to(dev),
                             variable_index_mask= feature.variable_index_mask.to(dev),
                             labels=feature.labels.to(dev),
                             return_dict=True, is_eval=True).all_logits ##List of (batch_size, num_combinations/num_m0, num_labels, 2, 2)
                num_variables = feature.num_variables.cpu().numpy().tolist()  # batch_size
                max_num_variable = max(num_variables)
                # (batch_size, max_height, num_combinations/num_m0, num_labels, 2, 2)
                pad_all_logits = pad_sequence([logits.permute(1,0,2,3,4) for logits in all_logits],
                                              padding_value=-np.inf).permute(2,1,0,3,4,5)
                _, best_final_label = pad_all_logits.max(dim=-1)  ## batch_size, height, num_combinations/num_m0, num_labels, 2
                ## TODO: if both 1 for stop_id=0 and 1 for stop_id = 1, what to do?..find sum===2, then use for loop to modify
                all_transform_predictions = get_transform_labels_from_batch_labels(best_final_label, max_num_variable=max_num_variable, constant_values=constant_values, add_empty_transform_labels=True)
                all_transform_predictions = check_multiple_stops_and_remove_logit(all_transform_predictions)
                ## check if there are multiple stops
                all_transform_labels = get_transform_labels_from_batch_labels(feature.labels, max_num_variable=max_num_variable, constant_values=constant_values)

                predictions.extend(all_transform_predictions)
                labels.extend(all_transform_labels)
    corr = 0
    num_label_step_corr = Counter()
    num_label_step_total = Counter()
    insts = valid_dataloader.dataset.insts
    for inst_predictions, inst_labels in zip(predictions, labels):
        num_label_step_total[len(inst_labels)] += 1
        if len(inst_predictions) != len(inst_labels):
            continue
        is_correct = True
        for prediction_step, label_step in zip(inst_predictions, inst_labels):
            if prediction_step != label_step:
                is_correct = False
                break
        if is_correct:
            num_label_step_corr[len(inst_labels)] += 1
            corr += 1
    total = len(labels)
    acc = corr*1.0/total
    print(f"[Info] Acc.:{acc*100:.2f} ", flush=True)

    ##value accuarcy
    val_corr = 0
    num_label_step_val_corr = Counter()
    err = []
    for inst_predictions, inst_labels, inst in zip(predictions, labels, insts):
        num_list = inst["num_list"]
        is_value_corr, predict_value, gold_value, pred_ground_equation, gold_ground_equation = is_value_correct(inst_predictions, inst_labels, num_list, num_constant=constant_num, constant_values=constant_values,
                                                                                                                consider_multiple_m0=True, use_parallel_equations=True)
        val_corr += 1 if is_value_corr else 0
        if is_value_corr:
            num_label_step_val_corr[len(inst_labels)] += 1
            corr += 1
        else:
            err.append(inst)
        inst["predict_value"] = predict_value
        inst["gold_value"] = gold_value
        inst['pred_ground_equation'] = pred_ground_equation
        inst['gold_ground_equation'] = gold_ground_equation
    val_acc = val_corr*1.0 / total
    print(f"[Info] val Acc.:{val_acc * 100:.2f} ", flush=True)
    for key in num_label_step_total:
        curr_corr = num_label_step_corr[key]
        curr_val_corr = num_label_step_val_corr[key]
        curr_total = num_label_step_total[key]
        print(f"[Info] step num: {key} Acc.:{curr_corr*1.0/curr_total * 100:.2f} ({curr_corr}/{curr_total}) val acc: {curr_val_corr*1.0/curr_total * 100:.2f} ({curr_val_corr}/{curr_total})", flush=True)
    if res_file is not None:
        write_data(file=res_file, data=insts)
    if err_file is not None:
        write_data(file=err_file, data=err)
    return val_acc

def main():
    parser = argparse.ArgumentParser(description="classificaton")
    opt = parse_arguments(parser)
    set_seed(opt)
    conf = Config(opt)

    bert_model_name = conf.bert_model_name if conf.bert_folder == "" else f"{conf.bert_folder}/{conf.bert_model_name}"
    ## update to latest type classification
    num_labels = 6

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

    if conf.add_new_token:
        print(f"[INFO] Adding new tokens <NUM> for numbering purpose, before: {len(tokenizer)}")
        tokenizer.add_tokens('<NUM>', special_tokens=True)

    if conf.use_constant:
        constant2id = {"1": 0, "PI": 1}
        constant_values = [1.0, 3.14]
        constant_number = len(constant_values)
    else:
        constant2id = None
        constant_values = None
        constant_number = 0

    # Read dataset
    if opt.mode == "train":
        print("[Data Info] Reading training data", flush=True)
        dataset = UniversalDataset(file=conf.train_file, tokenizer=tokenizer, number=conf.train_num, filtered_steps=opt.filtered_steps,
                                   constant2id=constant2id, constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                                   use_incremental_labeling=bool(conf.consider_multiple_m0), add_new_token=bool(conf.add_new_token))
        print("[Data Info] Reading validation data", flush=True)
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, number=conf.dev_num, filtered_steps=opt.filtered_steps,
                                        constant2id=constant2id, constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                                   use_incremental_labeling=bool(conf.consider_multiple_m0), add_new_token=bool(conf.add_new_token))


        # Prepare data loader
        print("[Data Info] Loading training data", flush=True)
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle_train_data, num_workers=conf.num_workers, collate_fn=dataset.collate_parallel)
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_parallel)


        # Train the model
        model = train(conf, train_dataloader,
                      num_epochs= conf.num_epochs,
                      bert_model_name = bert_model_name,
                      valid_dataloader = valid_dataloader,
                      dev=conf.device, tokenizer=tokenizer, num_labels=num_labels,
                      constant_values=constant_values)
        evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16), constant_values=constant_values)
    else:
        print(f"Testing the model now.")
        model = ParallelModel.from_pretrained(f"model_files/{conf.model_folder}",
                                               num_labels=num_labels,
                                               diff_param_for_height=conf.diff_param_for_height,
                                               height = conf.height,
                                               constant_num = constant_number).to(conf.device)
        print("[Data Info] Reading test data", flush=True)
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, number=conf.dev_num, filtered_steps=opt.filtered_steps,
                                        constant2id=constant2id, constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                                        use_incremental_labeling=bool(conf.consider_multiple_m0), add_new_token=bool(conf.add_new_token))
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0,
                                      collate_fn=eval_dataset.collate_parallel)
        os.makedirs("results", exist_ok=True)
        res_file= f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16), constant_values=constant_values, res_file=res_file, err_file=err_file)

if __name__ == "__main__":
    main()

