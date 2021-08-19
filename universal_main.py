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


    parser.add_argument('--train_file', type=str, default="data/complex/mwp_processed_train.json")
    parser.add_argument('--dev_file', type=str, default="data/complex/mwp_processed_test.json")

    parser.add_argument('--filtered_steps', default=None, nargs='+', help="some heights to filter")

    # model
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--model_folder', type=str, default="math_solver", help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="hfl", help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="chinese-roberta-wwm-ext",
                        help="The bert model name to used")
    parser.add_argument('--diff_param_for_height', type=int, default=0, choices=[0,1])


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
          dev: torch.device, tokenizer: PreTrainedTokenizer, valid_dataloader: DataLoader = None):

    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * num_epochs)

    model = UniversalModel.from_pretrained(bert_model_name, diff_param_for_height=config.diff_param_for_height, num_labels=num_labels).to(dev)
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
            performance = evaluate(valid_dataloader, model, dev, fp16=bool(config.fp16))
            if performance > best_performance:
                print(f"[Model Info] Saving the best model... with performance {performance}..")
                best_performance = performance
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(f"model_files/{config.model_folder}")
                tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    print(f"[Model Info] Best validation performance: {best_performance}")
    model = UniversalModel.from_pretrained(f"model_files/{config.model_folder}", diff_param_for_height=config.diff_param_for_height).to(dev)
    if config.fp16:
        model.half()
        model.save_pretrained(f"model_files/{config.model_folder}")
        tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    return model

def evaluate(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device, fp16:bool) -> float:
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                all_logits = model(input_ids=feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                             token_type_ids=feature.token_type_ids.to(dev),
                             variable_indexs_start=feature.variable_indexs_start.to(dev),
                             variable_indexs_end=feature.variable_indexs_end.to(dev),
                             num_variables = feature.num_variables.to(dev),
                             variable_index_mask= feature.variable_index_mask.to(dev),
                             labels=feature.labels.to(dev),
                             return_dict=True, is_eval=True).all_logits
                batch_size, max_num_variable = feature.variable_indexs_start.size()
                num_var_range = torch.arange(0, max_num_variable, device=feature.variable_indexs_start.device)
                combination = torch.combinations(num_var_range, r=2, with_replacement=False)  ##number_of_combinations x 2
                num_combinations, _ = combination.size()
                batched_prediction = [[] for _ in range(batch_size)]
                for k, logits in enumerate(all_logits):
                    best_temp_score, best_temp_label = logits.max(dim=-1)  ## batch_size, num_combinations
                    best_m0_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
                    best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size
                    if k == 0:
                        # batch_size x 2
                        best_comb_var_idxs = torch.gather(combination.unsqueeze(0).expand(batch_size, num_combinations, 2), 1,
                                     best_comb.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 2).to(feature.variable_indexs_start.device)).squeeze(1)
                    else:
                        # batch_size
                        best_comb_var_idxs = best_comb
                    best_comb_var_idxs = best_comb_var_idxs.cpu().numpy()
                    best_labels = best_label.cpu().numpy()
                    for b_idx, (best_comb_idx, best_label) in enumerate(zip(best_comb_var_idxs, best_labels)): ## within each instances:
                        if isinstance(best_comb_idx, np.int64):
                            right = best_comb_idx
                            left = -1
                        else:
                            left, right = best_comb_idx
                        curr_label = [left, right, best_label]
                        batched_prediction[b_idx].append(curr_label)
                ## post process remve extra
                for b, inst_predictions in enumerate(batched_prediction):
                    for p, prediction_step in enumerate(inst_predictions):
                        left, right, op_id = prediction_step
                        if right == max_num_variable:
                            batched_prediction[b] = batched_prediction[b][:p]
                            break
                batched_labels = feature.labels.cpu().numpy().tolist()
                for b, inst_labels in enumerate(batched_labels):
                    for p, label_step in enumerate(inst_labels):
                        left, right, op_id = label_step
                        if right == max_num_variable:
                            batched_labels[b] = batched_labels[b][:p]
                            break

                predictions.extend(batched_prediction)
                labels.extend(batched_labels)
    corr = 0
    num_label_step_corr = Counter()
    num_label_step_total = Counter()
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
    for key in num_label_step_total:
        curr_corr = num_label_step_corr[key]
        curr_total = num_label_step_total[key]
        print(f"[Info] step num: {key} Acc.:{curr_corr*1.0/curr_total * 100:.2f} ({curr_corr}/{curr_total})", flush=True)
    return acc

def main():
    parser = argparse.ArgumentParser(description="classificaton")
    opt = parse_arguments(parser)
    set_seed(opt)
    conf = Config(opt)

    bert_model_name = conf.bert_model_name if conf.bert_folder == "" else f"{conf.bert_folder}/{conf.bert_model_name}"
    ## update to latest type classification
    num_labels = 6

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)


    # Read dataset
    if opt.mode == "train":
        print("[Data Info] Reading training data", flush=True)
        dataset = UniversalDataset(file=conf.train_file, tokenizer=tokenizer, number=conf.train_num, filtered_steps=opt.filtered_steps)
        print("[Data Info] Reading validation data", flush=True)
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, number=conf.dev_num, filtered_steps=opt.filtered_steps)


        # Prepare data loader
        print("[Data Info] Loading training data", flush=True)
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle_train_data, num_workers=conf.num_workers, collate_fn=dataset.collate_function)
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_function)


        # Train the model
        model = train(conf, train_dataloader,
                      num_epochs= conf.num_epochs,
                      bert_model_name= bert_model_name,
                      valid_dataloader= valid_dataloader,
                      dev=conf.device, tokenizer=tokenizer, num_labels=num_labels)
        evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16))
    else:
        print(f"Testing the model now.")
        model = UniversalModel.from_pretrained(f"model_files/{conf.model_folder}", num_labels=num_labels, diff_param_for_height=conf.diff_param_for_height).to(conf.device)
        print("[Data Info] Reading test data", flush=True)
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, number=conf.dev_num, filtered_steps=opt.filtered_steps)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0,
                                      collate_fn=eval_dataset.collate_function)
        res_file= f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16))

if __name__ == "__main__":
    main()

