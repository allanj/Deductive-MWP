from src.data.universal_dataset import UniversalDataset, uni_labels
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
from src.model.universal_model import UniversalModel
from collections import Counter
from src.eval.utils import is_value_correct
from typing import List

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
    parser.add_argument('--train_num', type=int, default=-1, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=-1, help="The number of development data, -1 means all data")


    parser.add_argument('--train_file', type=str, default="data/math23k/train23k_processed_nodup.json")
    parser.add_argument('--dev_file', type=str, default="data/math23k/test23k_processed_nodup.json")

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
    model = UniversalModel.from_pretrained(bert_model_name,
                                           diff_param_for_height=config.diff_param_for_height,
                                           num_labels=num_labels,
                                           height=config.height,
                                           constant_num=constant_num,
                                           add_replacement=bool(config.add_replacement),
                                           consider_multiple_m0=bool(config.consider_multiple_m0)).to(dev)
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

    best_performance = -1
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
                             const_start = feature.const_start.to(dev),
                             const_end=feature.const_end.to(dev),
                             num_variables = feature.num_variables.to(dev),
                             variable_index_mask= feature.variable_index_mask.to(dev),
                             labels=feature.labels.to(dev), label_height_mask= feature.label_height_mask.to(dev),
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
            performance = evaluate(valid_dataloader, model, dev, fp16=bool(config.fp16), constant_values=constant_values,
                                   add_replacement=bool(config.add_replacement), consider_multiple_m0=bool(config.consider_multiple_m0))
            if performance > best_performance:
                print(f"[Model Info] Saving the best model... with performance {performance}..")
                best_performance = performance
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(f"model_files/{config.model_folder}")
                tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    print(f"[Model Info] Best validation performance: {best_performance}")
    model = UniversalModel.from_pretrained(f"model_files/{config.model_folder}",
                                           diff_param_for_height=config.diff_param_for_height,
                                           num_labels=num_labels,
                                           height=config.height,
                                           constant_num=constant_num,
                                           add_replacement=bool(config.add_replacement),
                                           consider_multiple_m0=bool(config.consider_multiple_m0)).to(dev)
    if config.fp16:
        model.half()
        model.save_pretrained(f"model_files/{config.model_folder}")
        tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    return model

def get_batched_prediction_consider_multiple_m0(feature, all_logits: torch.FloatTensor, constant_num: int, add_replacement: bool = False):
    batch_size, max_num_variable = feature.variable_indexs_start.size()
    device = feature.variable_indexs_start.device
    batched_prediction = [[] for _ in range(batch_size)]
    for k, logits in enumerate(all_logits):
        current_max_num_variable = max_num_variable + constant_num + k
        num_var_range = torch.arange(0, current_max_num_variable, device=feature.variable_indexs_start.device)
        combination = torch.combinations(num_var_range, r=2, with_replacement=add_replacement)  ##number_of_combinations x 2
        num_combinations, _ = combination.size()

        best_temp_logits, best_temp_stop_label = logits.max(dim=-1)  ## batch_size, num_combinations/num_m0, num_labels
        best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)  ## batch_size, num_combinations
        best_m0_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
        best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size
        b_idxs = [bidx for bidx in range(batch_size)]
        best_stop_label = best_temp_stop_label[b_idxs, best_comb, best_label] ## batch size

        # batch_size x 2
        best_comb_var_idxs = torch.gather(combination.unsqueeze(0).expand(batch_size, num_combinations, 2), 1,
                                          best_comb.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 2).to(device)).squeeze(1)
        best_comb_var_idxs = best_comb_var_idxs.cpu().numpy()
        best_labels = best_label.cpu().numpy()
        curr_best_stop_labels = best_stop_label.cpu().numpy()
        for b_idx, (best_comb_idx, best_label, stop_label) in enumerate(zip(best_comb_var_idxs, best_labels, curr_best_stop_labels)):  ## within each instances:
            left, right = best_comb_idx
            curr_label = [left, right, best_label, stop_label]
            batched_prediction[b_idx].append(curr_label)
    return batched_prediction


def get_batched_prediction(feature, all_logits: torch.FloatTensor, constant_num: int, add_replacement: bool = False):
    batch_size, max_num_variable = feature.variable_indexs_start.size()
    max_num_variable = max_num_variable + constant_num
    num_var_range = torch.arange(0, max_num_variable, device=feature.variable_indexs_start.device)
    combination = torch.combinations(num_var_range, r=2, with_replacement=add_replacement)  ##number_of_combinations x 2
    num_combinations, _ = combination.size()
    batched_prediction = [[] for _ in range(batch_size)]
    for k, logits in enumerate(all_logits):
        best_temp_logits, best_temp_stop_label = logits.max(dim=-1)  ## batch_size, num_combinations/num_m0, num_labels
        best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)  ## batch_size, num_combinations
        best_m0_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
        best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size
        b_idxs = [bidx for bidx in range(batch_size)]
        best_stop_label = best_temp_stop_label[b_idxs, best_comb, best_label]
        if k == 0:
            # batch_size x 2
            best_comb_var_idxs = torch.gather(combination.unsqueeze(0).expand(batch_size, num_combinations, 2), 1,
                                              best_comb.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 2).to(
                                                  feature.variable_indexs_start.device)).squeeze(1)
        else:
            # batch_size
            best_comb_var_idxs = best_comb
        best_comb_var_idxs = best_comb_var_idxs.cpu().numpy()
        best_labels = best_label.cpu().numpy()
        curr_best_stop_labels = best_stop_label.cpu().numpy()
        for b_idx, (best_comb_idx, best_label, stop_label) in enumerate(
                zip(best_comb_var_idxs, best_labels, curr_best_stop_labels)):  ## within each instances:
            if isinstance(best_comb_idx, np.int64):
                right = best_comb_idx
                left = -1
            else:
                left, right = best_comb_idx
            curr_label = [left, right, best_label, stop_label]
            batched_prediction[b_idx].append(curr_label)
    return batched_prediction

def evaluate(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device, fp16:bool, constant_values: List,
             add_replacement: bool = False, consider_multiple_m0: bool = False, res_file: str= None, err_file:str = None) -> float:
    model.eval()
    predictions = []
    labels = []
    constant_num = len(constant_values) if constant_values else 0
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                all_logits = model(input_ids=feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                             token_type_ids=feature.token_type_ids.to(dev),
                             variable_indexs_start=feature.variable_indexs_start.to(dev),
                             variable_indexs_end=feature.variable_indexs_end.to(dev),
                             const_start = feature.const_start.to(dev),
                             const_end=feature.const_end.to(dev),
                             num_variables = feature.num_variables.to(dev),
                             variable_index_mask= feature.variable_index_mask.to(dev),
                             labels=feature.labels.to(dev), label_height_mask= feature.label_height_mask.to(dev),
                             return_dict=True, is_eval=True).all_logits
                batched_prediction = get_batched_prediction(feature=feature, all_logits=all_logits, constant_num=constant_num, add_replacement=add_replacement) \
                    if not consider_multiple_m0 else get_batched_prediction_consider_multiple_m0(feature=feature, all_logits=all_logits, constant_num=constant_num, add_replacement=add_replacement)
                ## post process remve extra
                for b, inst_predictions in enumerate(batched_prediction):
                    for p, prediction_step in enumerate(inst_predictions):
                        left, right, op_id, stop_id = prediction_step
                        if stop_id == 1:
                            batched_prediction[b] = batched_prediction[b][:(p+1)]
                            break
                batched_labels = feature.labels.cpu().numpy().tolist()
                for b, inst_labels in enumerate(batched_labels):
                    for p, label_step in enumerate(inst_labels):
                        left, right, op_id, stop_id = label_step
                        if stop_id == 1:
                            batched_labels[b] = batched_labels[b][:(p+1)]
                            break

                predictions.extend(batched_prediction)
                labels.extend(batched_labels)
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
        is_value_corr, predict_value, gold_value, pred_ground_equation, gold_ground_equation = is_value_correct(inst_predictions, inst_labels, num_list, num_constant=constant_num, constant_values=constant_values, consider_multiple_m0=consider_multiple_m0)
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
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle_train_data, num_workers=conf.num_workers, collate_fn=dataset.collate_function)
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_function)


        # Train the model
        model = train(conf, train_dataloader,
                      num_epochs= conf.num_epochs,
                      bert_model_name = bert_model_name,
                      valid_dataloader = valid_dataloader,
                      dev=conf.device, tokenizer=tokenizer, num_labels=num_labels,
                      constant_values=constant_values)
        evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16), constant_values=constant_values, add_replacement=bool(conf.add_replacement), consider_multiple_m0=bool(conf.consider_multiple_m0))
    else:
        print(f"Testing the model now.")
        model = UniversalModel.from_pretrained(f"model_files/{conf.model_folder}",
                                               num_labels=num_labels,
                                               diff_param_for_height=conf.diff_param_for_height,
                                               height = conf.height,
                                               constant_num = constant_number,
                                            add_replacement=bool(conf.add_replacement), consider_multiple_m0=conf.consider_multiple_m0).to(conf.device)
        print("[Data Info] Reading test data", flush=True)
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, number=conf.dev_num, filtered_steps=opt.filtered_steps,
                                        constant2id=constant2id, constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                                        use_incremental_labeling=bool(conf.consider_multiple_m0), add_new_token=bool(conf.add_new_token))
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0,
                                      collate_fn=eval_dataset.collate_function)
        os.makedirs("results", exist_ok=True)
        res_file= f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16), constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                 consider_multiple_m0=bool(conf.consider_multiple_m0), res_file=res_file, err_file=err_file)

if __name__ == "__main__":
    main()

