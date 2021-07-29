from src.data.math_dataset import QuestionDataset, measure_two_expr
from src.data.four_variable_dataset import FourVariableDataset
from src.config import Config
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, PreTrainedTokenizer
from tqdm import tqdm
import argparse
from src.utils import get_optimizers, write_data
import torch
import torch.nn as nn
import numpy as np
import os
import random
from src.model.scoring_model import ScoringModel


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed_all(args.seed)

def parse_arguments(parser:argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], help="GPU/CPU devices")
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--shuffle_train_data', type=int, default=1, choices=[0, 1], help="shuffle the training data or not")
    parser.add_argument('--max_seq_length', type=int, default=200, choices=range(50, 201), help="maximum sequence length")
    parser.add_argument('--train_num', type=int, default=-1, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=-1, help="The number of development data, -1 means all data")

    parser.add_argument('--max_length_direction', type=str, default="left_to_right", choices=["left_to_right", "right_to_left"], help="sentence max length selection from left to right or right to left")
    parser.add_argument('--filtered_max_length', type=int, default=-1, help="filter both training and dev data that have length longer than this parameter. -1 means no filtering")

    parser.add_argument('--train_file', type=str, default="data/simple_cases_train_all.json")
    parser.add_argument('--dev_file', type=str, default="data/simple_cases_test_all.json")
    parser.add_argument('--fv_train_file', type=str, default="data/fv_train_updated.json")
    parser.add_argument('--fv_dev_file', type=str, default="data/fv_test_updated.json")

    parser.add_argument('--four_variables', type=int, default=0, choices=[0, 1], help="random seed")
    # model
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--model_folder', type=str, default="math_solver", help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="hfl", help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="chinese-roberta-wwm-ext",
                        help="The bert model name to used")

    ## data and model
    parser.add_argument('--insert_m0_string', type=int, default=1, choices=[0,1], help="whether we insert the m0 string into the model")

    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="learning rate of the AdamW optimizer")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate of the AdamW optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")
    parser.add_argument('--num_epochs', type=int, default=20, help="The number of epochs to run")
    parser.add_argument('--temperature', type=float, default=1.0, help="The temperature during the training")
    parser.add_argument('--fp16', type=int, default=0, choices=[0,1], help="using fp16 to train the model")

    parser.add_argument('--use_binary', type=int, default=1, choices=[0, 1], help="using fp16 to train the model")
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
          idx2labels=None, pretty_idx2labels=None, fv_train_loader:DataLoader = None, fv_valid_loader: DataLoader = None):

    gradient_accumulation_steps = 1
    t_total = int((len(train_dataloader) + len(fv_train_loader)) // gradient_accumulation_steps * num_epochs)

    model = ScoringModel.from_pretrained(bert_model_name, num_labels=num_labels).to(dev)
    if config.parallel:
        model = nn.DataParallel(model)

    scaler = None
    if config.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config.fp16))

    optimizer, scheduler = get_optimizers(config, model, t_total)
    model.zero_grad()

    best_performance = -1
    os.makedirs(f"model_files/{config.model_folder}", exist_ok=True)
    total_num_batches = len(train_dataloader) + len(fv_train_loader)
    orders = [0] * len(train_dataloader) + [1] * len(fv_train_loader)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        three_var_iter = enumerate(train_dataloader, 1)
        fv_iter = enumerate(fv_train_loader, 1)
        random.shuffle(orders)
        for iter, dataset_idx in tqdm(enumerate(orders, 1), desc="--training batch", total=total_num_batches):

            if dataset_idx == 0:
                _, feature = next(three_var_iter)
                args = {
                    "dataset": feature.dataset, "input_ids": feature.input_ids.to(dev),
                    "attention_mask": feature.attention_mask.to(dev),
                    "sent_starts": feature.sent_starts.to(dev),
                    "sent_ends": feature.sent_ends.to(dev),
                    "labels": feature.label_id.to(dev),
                    "return_dict": True
                }
            else:
                _, feature = next(fv_iter)
                args = {
                    "dataset": feature.dataset, "input_ids": feature.input_ids.to(dev),
                    "attention_mask": feature.attention_mask.to(dev),
                    "sent_starts": feature.sent_starts.to(dev), "m0_sent_starts": feature.m0_sent_starts.to(dev),
                    "sent_ends": feature.sent_ends.to(dev),  "m0_sent_ends": feature.m0_sent_starts.to(dev),
                    "m0_operator_ids": feature.m0_operator_ids.to(dev),
                    "labels": feature.label_id.to(dev),
                    "return_dict": True
                }
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                loss = model(**args).loss.sum()
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
            performance = evaluate(valid_dataloader, model, dev, fp16=bool(config.fp16), idx2labels=idx2labels, pretty_idx2labels=pretty_idx2labels, use_binary=config.use_binary)
            fv_perf = evaluate_four_variable(fv_valid_loader, model, dev, fp16=bool(config.fp16))
            total_perf = (performance + fv_perf) / 2
            if total_perf > best_performance:
                print(f"[Model Info] Saving the best model... with average performance {total_perf}..")
                best_performance = total_perf
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(f"model_files/{config.model_folder}")
                tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    print(f"[Model Info] Best validation performance: {best_performance}")
    model = ScoringModel.from_pretrained(f"model_files/{config.model_folder}").to(dev)
    if config.fp16:
        model.half()
        model.save_pretrained(f"model_files/{config.model_folder}")
        tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    return model

def evaluate_four_variable(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device,
             fp16:bool):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                logits = model(dataset=feature.dataset, input_ids=feature.input_ids.to(dev),
                                           attention_mask=feature.attention_mask.to(dev),
                                           sent_starts=feature.sent_starts.to(dev), m0_sent_starts=feature.m0_sent_starts.to(dev),
                                           sent_ends=feature.sent_ends.to(dev), m0_sent_ends=feature.m0_sent_ends.to(dev), m0_operator_ids=feature.m0_operator_ids.to(dev)).logits
            logits = logits[:, :, :, 1] # batch_size, num_m0, 6 label scores
            temp_scores, temp_prediction = logits.max(dim=-1) # batch_size, num_m0 (the best label index for each m0)
            _, best_m0_idx = temp_scores.max(dim=-1) # batch_size
            # batch_size
            curr_best_label_id = np.squeeze(np.take_along_axis(temp_prediction, np.expand_dims(best_m0_idx.cpu().numpy(), axis=1), axis=1), axis=1)
            total = len(curr_best_label_id)
            for best_m0, best_label_id in zip(best_m0_idx.cpu().numpy(), curr_best_label_id):
                predictions.append((best_m0, best_label_id))
            label_id = feature.label_id.cpu().numpy() # batch_size, num_m0, 6 label: 1,0
            k = 0
            t = np.where(label_id==1)
            for _, m, l in zip(t[0], t[1], t[2]):
                labels.append((m, l))
                k += 1
            assert k == total

    m0_corr = 0
    corr = 0
    for pred_tuple, gold_tuple in zip(predictions, labels):
        if pred_tuple[0] == gold_tuple[0]:
            m0_corr += 1
            if pred_tuple[1] == gold_tuple[1]:
                corr += 1
    m0_acc = m0_corr * 1.0 / len(predictions) * 100
    acc = corr * 1.0 / len(predictions) * 100
    print(f"[Info]  m0_acc.:{m0_acc:.2f}, total acc: {acc:.2f} , total number: {len(predictions)}", flush=True)
    return acc / 100

def evaluate(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device,
             fp16:bool, eval_answer: bool = False ,
             result_file:str = None, err_file:str = None, idx2labels=None, pretty_idx2labels=None,
             use_binary: bool = False) -> float:
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                logits = model(dataset=feature.dataset, input_ids = feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                               sent_starts = feature.sent_starts.to(dev), sent_ends = feature.sent_ends.to(dev)).logits
            if use_binary:
                # logits: batch_size, num_labels, 2
                logits = logits[:, :, 1]
            _, current_prediction = logits.max(dim=-1)

            predictions.extend(current_prediction.cpu().numpy().tolist())
            label_id = feature.label_id.cpu().numpy()
            if use_binary:
                xs, ys = np.where(label_id==1)
                assert len(xs) == len(current_prediction)
                label_id = ys
            labels.extend(label_id)

    predictions = np.array(predictions)
    labels = np.array(labels)
    ret = predictions == labels
    if not eval_answer:
        acc = float(np.mean(ret))
    else:
        corr = 0
        for label_id, pred_id, inst in zip(labels, predictions, valid_dataloader.dataset.insts):
            if label_id == pred_id:
                corr +=1
            else:
                gold_expr_str = idx2labels[label_id]
                pred_expr_str = idx2labels[pred_id]
                if measure_two_expr(gold_expr_str, pred_expr_str, inst["nums"], inst["ans_idx"]):
                    corr +=1
        acc = corr*1.0 / len(predictions)
    print(f"[Info] Acc.:{acc*100:.2f} total number: {len(predictions)}", flush=True)
    if result_file is not None:
        res = []
        err = []
        for label_id, pred_id, inst in zip(labels, predictions, valid_dataloader.dataset.insts):
            gold_label = pretty_idx2labels[label_id]
            pred_label = pretty_idx2labels[pred_id]
            inst["prediction"] = pred_label
            inst["gold_label"] = gold_label
            if label_id != pred_id:
                gold_expr_str = idx2labels[label_id]
                pred_expr_str = idx2labels[pred_id]
                if eval_answer and measure_two_expr(gold_expr_str, pred_expr_str, inst["nums"], inst["ans_idx"]):
                    pass
                else:
                    err.append(inst)
            res.append(inst)
        write_data(file=result_file, data=res)
        write_data(file=err_file, data=err)
    return acc

def main():
    parser = argparse.ArgumentParser(description="classificaton")
    opt = parse_arguments(parser)
    set_seed(opt)
    conf = Config(opt)

    bert_model_name = conf.bert_model_name if conf.bert_folder == "" else f"{conf.bert_folder}/{conf.bert_model_name}"
    ## update to latest type classification
    num_labels = 6 if not opt.four_variables else 24
    if opt.four_variables:
        from src.data.utils import pretty_labels as pretty_idx2labels
        from src.data.utils import fv_labels as idx2labels
    else:
        from src.data.math_dataset import pretty_labels as pretty_idx2labels
        from src.data.math_dataset import labels as idx2labels

    tokenizer = RobertaTokenizerFast.from_pretrained(bert_model_name)


    # Read dataset
    if opt.mode == "train":
        print("[Data Info] Reading training data", flush=True)
        dataset = QuestionDataset(file=conf.train_file, tokenizer=tokenizer, number=conf.train_num, use_four_variables=opt.four_variables,
                                  use_binary=opt.use_binary, use_ans_string=opt.insert_m0_string)
        print("[Data Info] Reading validation data", flush=True)
        eval_dataset = QuestionDataset(file=conf.dev_file, tokenizer=tokenizer, number=conf.dev_num, use_four_variables=opt.four_variables,
                                  use_binary=opt.use_binary, use_ans_string=opt.insert_m0_string)


        # Prepare data loader
        print("[Data Info] Loading training data", flush=True)
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle_train_data, num_workers=conf.num_workers, collate_fn=dataset.collate_function)
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_function)


        fv_train_dataset = FourVariableDataset(file=opt.fv_train_file, tokenizer=tokenizer, number=conf.train_num, insert_m0_string=opt.insert_m0_string)
        fv_eval_dataset = FourVariableDataset(file=opt.fv_dev_file, tokenizer=tokenizer, number=conf.dev_num, insert_m0_string=opt.insert_m0_string)
        print("[Data Info] Loading fv training data", flush=True)
        fv_train_dataloader = DataLoader(fv_train_dataset, batch_size=conf.batch_size, shuffle=conf.shuffle_train_data,
                                      num_workers=conf.num_workers, collate_fn=fv_train_dataset.collate_function)
        print("[Data Info] Loading fv validation data", flush=True)
        fv_valid_dataloader = DataLoader(fv_eval_dataset, batch_size=conf.batch_size, shuffle=False,
                                      num_workers=conf.num_workers, collate_fn=fv_eval_dataset.collate_function)

        # Train the model
        model = train(conf, train_dataloader,
                      num_epochs= conf.num_epochs,
                      bert_model_name= bert_model_name,
                      valid_dataloader= valid_dataloader,
                      dev=conf.device, tokenizer=tokenizer, num_labels=num_labels,
                      idx2labels=idx2labels,
                      pretty_idx2labels=pretty_idx2labels,
                      fv_train_loader = fv_train_dataloader,
                      fv_valid_loader = fv_valid_dataloader)
        evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16), use_binary=opt.use_binary)
    else:
        print(f"Testing the model now.")
        model = ScoringModel.from_pretrained(f"model_files/{conf.model_folder}", num_labels=num_labels).to(conf.device)
        print("[Data Info] Reading test data", flush=True)
        eval_dataset = QuestionDataset(file=conf.dev_file, tokenizer=tokenizer, number=conf.dev_num,
                                  use_binary=opt.use_binary, use_ans_string=opt.insert_m0_string)
        fv_eval_dataset = FourVariableDataset(file=opt.fv_dev_file, tokenizer=tokenizer, number=conf.dev_num, insert_m0_string=opt.insert_m0_string)
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0,
                                      collate_fn=eval_dataset.collate_function)
        fv_valid_dataloader = DataLoader(fv_eval_dataset, batch_size=conf.batch_size, shuffle=False,
                                         num_workers=0, collate_fn=fv_eval_dataset.collate_function)
        res_file= f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16), eval_answer=False,
                 # result_file=res_file, err_file=err_file,
                      idx2labels=idx2labels,
                      pretty_idx2labels=pretty_idx2labels)
        evaluate_four_variable(fv_valid_dataloader, model, conf.device, fp16=bool(conf.fp16))

if __name__ == "__main__":
    main()

