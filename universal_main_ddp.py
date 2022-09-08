from src.data.universal_dataset import UniversalDataset
from src.config import Config
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, PreTrainedTokenizer, RobertaTokenizerFast, XLMRobertaTokenizerFast
from tqdm import tqdm
import argparse
from src.utils import get_optimizers, write_data
import torch
import torch.nn as nn
import numpy as np
import os
import random
from src.model.universal_model import UniversalModel
from src.model.universal_model_roberta import UniversalModel_Roberta
from src.model.universal_model_bert import UniversalModel_Bert
from src.model.universal_model_xlmroberta import UniversalModel_XLMRoberta
from collections import Counter
from src.eval.utils import is_value_correct
from typing import List, Tuple
import logging
from transformers import set_seed
from accelerate import Accelerator
from accelerate.utils import pad_across_processes
from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
if accelerator.is_local_main_process:
    logger.setLevel(logging.INFO)  ## so only print something in main process
else:
    logger.setLevel(logging.WARNING)

class_name_2_model = {
        "bert-base-cased": UniversalModel_Bert,
        "roberta-base": UniversalModel_Roberta,
        "bert-base-multilingual-cased": UniversalModel_Bert,
        'bert-base-chinese': UniversalModel,
        "xlm-roberta-base": UniversalModel_XLMRoberta,
        'hfl/chinese-bert-wwm-ext': UniversalModel,
        'hfl/chinese-roberta-wwm-ext': UniversalModel,
    }

def parse_arguments(parser:argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'], help="GPU/CPU devices")
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--train_num', type=int, default=-1, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=-1, help="The number of development data, -1 means all data")
    parser.add_argument('--test_num', type=int, default=-1, help="The number of development data, -1 means all data")


    parser.add_argument('--train_file', type=str, default="data/math23k/train23k_processed_nodup.json")
    parser.add_argument('--dev_file', type=str, default="data/math23k/valid23k_processed_nodup.json")
    parser.add_argument('--test_file', type=str, default="data/math23k/test23k_processed_nodup.json")
    # parser.add_argument('--train_file', type=str, default="data/mawps-single/mawps_train_nodup.json")
    # parser.add_argument('--dev_file', type=str, default="data/mawps-single/mawps_test_nodup.json")

    parser.add_argument('--train_filtered_steps', default=None, nargs='+', help="some heights to filter")
    parser.add_argument('--test_filtered_steps', default=None, nargs='+', help="some heights to filter")
    parser.add_argument('--use_constant', default=1, type=int, choices=[0,1], help="whether to use constant 1 and pi")

    parser.add_argument('--add_replacement', default=1, type=int, choices=[0,1], help = "use replacement when computing combinations")

    # model
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--model_folder', type=str, default="math_solver", help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="", help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="chinese-roberta-wwm-ext",
                        help="The bert model name to used")
    # parser.add_argument('--bert_folder', type=str, default="", help="The folder name that contains the BERT model")
    # parser.add_argument('--bert_model_name', type=str, default="roberta-base",
    #                     help="The bert model name to used")
    parser.add_argument('--height', type=int, default=10, help="the model height")
    parser.add_argument('--train_max_height', type=int, default=100, help="the maximum height for training data")
    parser.add_argument('--consider_multiple_m0', type=int, default=1, help="whether or not to consider multiple m0")

    parser.add_argument('--var_update_mode', type=str, default="gru", help="variable update mode")

    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="learning rate of the AdamW optimizer")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate of the AdamW optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")
    parser.add_argument('--num_epochs', type=int, default=20, help="The number of epochs to run")
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
        logger.info(f"{k} = {args.__dict__[k]}")
    return args


def train(config: Config, train_dataloader: DataLoader, num_epochs: int,
          bert_model_name: str, num_labels: int,
          dev: torch.device, tokenizer: PreTrainedTokenizer, valid_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
          constant_values: List = None, res_file:str = None, error_file:str = None):

    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * num_epochs)
    t_total = int(t_total // accelerator.num_processes)

    constant_num = len(constant_values) if constant_values else 0
    MODEL_CLASS = class_name_2_model[bert_model_name]
    model = MODEL_CLASS.from_pretrained(bert_model_name,
                                           num_labels=num_labels,
                                           height=config.height,
                                           constant_num=constant_num,
                                           add_replacement=bool(config.add_replacement),
                                           consider_multiple_m0=bool(config.consider_multiple_m0),
                                            var_update_mode=config.var_update_mode, return_dict=True)

    optimizer, scheduler = get_optimizers(config, model, t_total)
    model.zero_grad()

    best_equ_acc = -1
    best_val_acc_performance = -1
    os.makedirs(f"model_files/{config.model_folder}", exist_ok=True)

    model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader, scheduler)
    if test_dataloader is not None:
        test_dataloader = accelerator.prepare(test_dataloader)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for iter, feature in tqdm(enumerate(train_dataloader, 1), desc="--training batch", total=len(train_dataloader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                loss = model(**feature._asdict()).loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if iter % 1000 == 0:
                logger.info(f"epoch: {epoch}, iteration: {iter}, current mean loss: {total_loss/iter:.2f}")
        logger.info(f"Finish epoch: {epoch}, loss: {total_loss:.2f}, mean loss: {total_loss/len(train_dataloader):.2f}")
        if valid_dataloader is not None:
            equ_acc, val_acc_performance = evaluate(valid_dataloader, model, dev, uni_labels=config.uni_labels, fp16=bool(config.fp16), constant_values=constant_values,
                                   add_replacement=bool(config.add_replacement), consider_multiple_m0=bool(config.consider_multiple_m0))
            test_equ_acc, test_val_acc = -1, -1
            if test_dataloader is not None:
                test_equ_acc, test_val_acc = evaluate(test_dataloader, model, dev, uni_labels=config.uni_labels, fp16=bool(config.fp16), constant_values=constant_values,
                                       add_replacement=bool(config.add_replacement), consider_multiple_m0=bool(config.consider_multiple_m0),
                         res_file=res_file, err_file=error_file)
            if val_acc_performance > best_val_acc_performance:
                logger.info(
                    f"[Model Info] Saving the best model with best valid val acc {val_acc_performance:.6f} at epoch {epoch} ("
                    f"valid_equ: {equ_acc:.6f}, valid_val: {val_acc_performance:.6f}"
                    f" test_equ: {test_equ_acc:.6f}, test_val: {test_val_acc:.6f}"
                    f")")
                best_val_acc_performance = val_acc_performance
                best_equ_acc = equ_acc
                if accelerator.is_local_main_process:
                    model_to_save = accelerator.unwrap_model(model)
                    model_to_save.save_pretrained(f"model_files/{config.model_folder}")
                    tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    logger.info(f"[Model Info] Best validation value performance: {best_val_acc_performance} (best_equ_acc: {best_equ_acc})")

def get_batched_prediction_consider_multiple_m0(feature, all_logits: torch.FloatTensor, constant_num: int, add_replacement: bool = False):
    batch_size, max_num_variable = feature.variable_indexs_start.size()
    device = feature.variable_indexs_start.device
    batched_predictions = []
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
        # batched_prediction[:,k, :] = torch.cat([best_comb_var_idxs, best_label.unsqueeze(-1), best_stop_label.unsqueeze(-1)], dim=-1)
        batched_predictions.append(torch.cat([best_comb_var_idxs, best_label.unsqueeze(-1), best_stop_label.unsqueeze(-1)], dim=-1))
        # best_comb_var_idxs = best_comb_var_idxs.cpu().numpy() ## batch_size x 2
        # best_labels = best_label.cpu().numpy() ## batch_size
        # curr_best_stop_labels = best_stop_label.cpu().numpy() ## batch_size
        # for b_idx, (best_comb_idx, best_label, stop_label) in enumerate(zip(best_comb_var_idxs, best_labels, curr_best_stop_labels)):  ## within each instances:
        #     left, right = best_comb_idx
        #     curr_label = [left.item(), right.item(), best_label.item(), stop_label.item()]
        #     batched_prediction[b_idx].append(curr_label)
    return torch.stack(batched_predictions, dim=1)


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
                right = best_comb_idx.item()
                left = -1
            else:
                left, right = best_comb_idx
                left, right = left.item(), right.item()
            curr_label = [left, right, best_label.item(), stop_label.item()]
            batched_prediction[b_idx].append(curr_label)
    return batched_prediction

def evaluate(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device, fp16:bool, constant_values: List, uni_labels:List,
             add_replacement: bool = False, consider_multiple_m0: bool = False, res_file: str= None, err_file:str = None) -> Tuple[float, float]:
    model.eval()
    predictions = []
    labels = []
    constant_num = len(constant_values) if constant_values else 0
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                all_logits = model(**feature._asdict(), is_eval=True).all_logits
                batched_prediction = get_batched_prediction(feature=feature, all_logits=all_logits, constant_num=constant_num, add_replacement=add_replacement) \
                    if not consider_multiple_m0 else get_batched_prediction_consider_multiple_m0(feature=feature, all_logits=all_logits, constant_num=constant_num, add_replacement=add_replacement)

                batched_prediction = accelerator.gather_for_metrics(batched_prediction)
                batched_labels = accelerator.pad_across_processes(feature.labels, dim=1, pad_index=-100)
                batched_labels = accelerator.gather_for_metrics(batched_labels)

                ## post process remve extra padding step
                batched_prediction = batched_prediction.cpu().numpy().tolist()
                for b, inst_predictions in enumerate(batched_prediction):
                    for p, prediction_step in enumerate(inst_predictions):
                        left, right, op_id, stop_id = prediction_step
                        if stop_id == 1:
                            batched_prediction[b] = batched_prediction[b][:(p+1)]
                            break
                batched_labels = batched_labels.cpu().numpy().tolist()
                for b, inst_labels in enumerate(batched_labels):
                    for p, label_step in enumerate(inst_labels):
                        left, right, op_id, stop_id = label_step
                        if stop_id == 1:
                            batched_labels[b] = batched_labels[b][:(p+1)]
                            break

                predictions.extend(batched_prediction)
                labels.extend(batched_labels)
    ## need to remove additional instances that are added by accelerator
    ## because of distributed training
    total_instance_num = len(valid_dataloader.dataset)
    logger.info(f"before removed: {len(predictions)}, after removed: {total_instance_num}")
    predictions = predictions[:total_instance_num]
    labels = labels[:total_instance_num]
    ##
    corr = 0
    num_label_step_corr = Counter()
    num_label_step_total = Counter()
    insts = valid_dataloader.dataset.insts
    number_instances_remove = valid_dataloader.dataset.number_instances_remove
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
    adjusted_total = total + number_instances_remove
    acc = corr*1.0/adjusted_total
    logger.info(f"[Info] Equation accuracy: {acc*100:.2f}%, total: {total}, corr: {corr}, adjusted_total: {adjusted_total}")

    ##value accuarcy
    val_corr = 0
    num_label_step_val_corr = Counter()
    err = []
    corr = 0
    for inst_predictions, inst_labels, inst in zip(predictions, labels, insts):
        num_list = inst["num_list"]
        is_value_corr, predict_value, gold_value, pred_ground_equation, gold_ground_equation = is_value_correct(inst_predictions, inst_labels, num_list, num_constant=constant_num, uni_labels=uni_labels, constant_values=constant_values, consider_multiple_m0=consider_multiple_m0)
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
    val_acc = val_corr * 1.0 / adjusted_total
    logger.info(f"[Info] Value accuracy: {val_acc * 100:.2f}%, total: {total}, corr: {val_corr}, adjusted_total: {adjusted_total}")
    for key in num_label_step_total:
        curr_corr = num_label_step_corr[key]
        curr_val_corr = num_label_step_val_corr[key]
        curr_total = num_label_step_total[key]
        logger.info(f"[Info] step num: {key} Acc.:{curr_corr*1.0/curr_total * 100:.2f} ({curr_corr}/{curr_total}) val acc: {curr_val_corr*1.0/curr_total * 100:.2f} ({curr_val_corr}/{curr_total})")
    if res_file is not None:
        write_data(file=res_file, data=insts)
    if err_file is not None:
        write_data(file=err_file, data=err)
    return acc, val_acc

def main():
    parser = argparse.ArgumentParser(description="classificaton")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)

    bert_model_name = conf.bert_model_name if conf.bert_folder == "" or conf.bert_folder=="none" else f"{conf.bert_folder}/{conf.bert_model_name}"
    class_name_2_tokenizer = {
        "bert-base-cased": BertTokenizerFast,
        "roberta-base": RobertaTokenizerFast,
        "bert-base-multilingual-cased": BertTokenizerFast,
        "xlm-roberta-base": XLMRobertaTokenizerFast,
        'bert-base-chinese': BertTokenizerFast,
        'hfl/chinese-bert-wwm-ext': BertTokenizerFast,
        'hfl/chinese-roberta-wwm-ext': BertTokenizerFast,
    }

    TOKENIZER_CLASS_NAME = class_name_2_tokenizer[bert_model_name]
    ## update to latest type classification

    tokenizer = TOKENIZER_CLASS_NAME.from_pretrained(bert_model_name)


    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    num_labels = 6
    conf.uni_labels = uni_labels
    if conf.use_constant:
        if "23k" in conf.train_file:
            conf.uni_labels = conf.uni_labels + ['^', '^_rev']
            num_labels = len(conf.uni_labels)
            constant2id = {"1": 0, "PI": 1}
            constant_values = [1.0, 3.14]
            constant_number = len(constant_values)
        elif "svamp" in conf.train_file:
            # ['0.01', '12.0', '1.0', '100.0', '0.1', '0.5', '3.0', '4.0', '7.0']
            constants = ['1.0', '0.1', '3.0', '5.0', '0.5', '12.0', '4.0', '60.0', '25.0', '0.01', '0.05', '2.0',
                         '10.0', '0.25', '8.0', '7.0', '100.0']
            constant2id = {c: idx for idx, c in enumerate(constants)}
            constant_values = [float(c) for c in constants]
            constant_number = len(constant_values)
        elif "mawps" in conf.train_file:
            constants = ['12.0', '1.0', '7.0', '60.0', '2.0', '5.0', '100.0', '8.0', '0.1', '0.5', '0.01', '25.0', '4.0', '3.0', '0.25']
            if conf.train_file.split(".")[-2][-1] in ["0", "1", "2", "3", "4", "5"]:  ## 5 fold trainning
                constants += ['10.0', '0.05']
            constant2id = {c: idx for idx, c in enumerate(constants)}
            constant_values = [float(c) for c in constants]
            constant_number = len(constant_values)
        elif "large_math" in conf.train_file:
            constants = ['5.0', '10.0', '2.0', '8.0', '30.0', '1.0', '6.0', '7.0', '12.0', '4.0', '31.0', '3.14', '3.0']
            constant2id = {c: idx for idx, c in enumerate(constants)}
            constant_values = [float(c) for c in constants]
            constant_number = len(constant_values)
        elif "MathQA" in conf.train_file:
            constants = ['100.0', '1.0', '2.0', '3.0', '4.0', '10.0', '1000.0', '60.0', '0.5', '3600.0', '12.0', '0.2778', '3.1416', '3.6', '0.25', '5.0', '6.0', '360.0', '52.0', '180.0']
            conf.uni_labels = conf.uni_labels + ['^', '^_rev']
            num_labels = len(conf.uni_labels)
            # constants = ['100.0', '1.0', '2.0', '3.0', '4.0', '10.0', '1000.0', '60.0', '0.5', '3600.0', '12.0', '0.2778', '3.1416']
            constant2id = {c: idx for idx, c in enumerate(constants)}
            constant_values = [float(c) for c in constants]
            constant_number = len(constant_values)
        else:
            constant2id = None
            constant_values = None
            constant_number = 0
    else:
        raise NotImplementedError
    logger.info(f"[Data Info] constant info: {constant2id}")


    # Read dataset
    if opt.mode == "train":
        logger.info("[Data Info] Reading training data")
        dataset = UniversalDataset(file=conf.train_file, tokenizer=tokenizer, uni_labels=conf.uni_labels, number=conf.train_num, filtered_steps=opt.train_filtered_steps,
                                   constant2id=constant2id, constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                                   use_incremental_labeling=bool(conf.consider_multiple_m0),
                                   data_max_height=opt.train_max_height, pretrained_model_name=bert_model_name)
        logger.info("[Data Info] Reading validation data")
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, uni_labels=conf.uni_labels, number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                        constant2id=constant2id, constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                                   use_incremental_labeling=bool(conf.consider_multiple_m0),
                                        data_max_height=conf.height, pretrained_model_name=bert_model_name)

        logger.info("[Data Info] Reading Testing data data")
        test_dataset = None
        if os.path.exists(conf.test_file):
            test_dataset = UniversalDataset(file=conf.test_file, tokenizer=tokenizer, uni_labels=conf.uni_labels,
                                            number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                            constant2id=constant2id, constant_values=constant_values,
                                            add_replacement=bool(conf.add_replacement),
                                            use_incremental_labeling=bool(conf.consider_multiple_m0),
                                            data_max_height=conf.height, pretrained_model_name=bert_model_name)
        logger.info(f"[Data Info] Training instances: {len(dataset)}, Validation instances: {len(eval_dataset)}")
        if test_dataset is not None:
            logger.info(f"[Data Info] Testing instances: {len(test_dataset)}")
        # Prepare data loader
        logger.info("[Data Info] Loading data")
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, collate_fn=dataset.collate_function)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_function)
        test_loader = None
        if test_dataset is not None:
            logger.info("[Data Info] Loading Test data")
            test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_function)

        res_file = f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        # Train the model
        train(conf, train_dataloader,
                  num_epochs = conf.num_epochs,
                  bert_model_name = bert_model_name,
                  valid_dataloader = valid_dataloader, test_dataloader=test_loader,
                  dev=conf.device, tokenizer=tokenizer, num_labels=num_labels,
                  constant_values=constant_values, res_file=res_file, error_file=err_file)
    else:
        logger.info(f"Testing the model now.")
        MODEL_CLASS = class_name_2_model[bert_model_name]
        model = MODEL_CLASS.from_pretrained(f"model_files/{conf.model_folder}",
                                               num_labels=num_labels,
                                               height = conf.height,
                                               constant_num = constant_number,
                                            add_replacement=bool(conf.add_replacement), consider_multiple_m0=conf.consider_multiple_m0,
                                            var_update_mode=conf.var_update_mode).to(conf.device)
        logger.info("[Data Info] Reading test data")
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, uni_labels=conf.uni_labels, number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                        constant2id=constant2id, constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                                        use_incremental_labeling=bool(conf.consider_multiple_m0), data_max_height=conf.height, pretrained_model_name=bert_model_name)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0,
                                      collate_fn=eval_dataset.collate_function)
        os.makedirs("results", exist_ok=True)
        res_file= f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        valid_dataloader, model = accelerator.prepare(valid_dataloader, model)
        evaluate(valid_dataloader, model, conf.device, uni_labels=conf.uni_labels, fp16=bool(conf.fp16), constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                 consider_multiple_m0=bool(conf.consider_multiple_m0), res_file=res_file, err_file=err_file)

if __name__ == "__main__":
    main()

