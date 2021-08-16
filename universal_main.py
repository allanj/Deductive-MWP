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
    parser.add_argument('--train_num', type=int, default=-1, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=-1, help="The number of development data, -1 means all data")


    parser.add_argument('--train_file', type=str, default="data/complex/mwp_processed_train.json")
    parser.add_argument('--dev_file', type=str, default="data/complex/mwp_processed_test.json")

    # model
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--model_folder', type=str, default="math_solver", help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="hfl", help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="chinese-roberta-wwm-ext",
                        help="The bert model name to used")


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

    model = UniversalModel.from_pretrained(bert_model_name, num_labels=num_labels).to(dev)
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
                             return_dict=True)
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
        # if valid_dataloader is not None:
        #     performance = evaluate(valid_dataloader, model, dev, fp16=bool(config.fp16), idx2labels=idx2labels, pretty_idx2labels=pretty_idx2labels, use_binary=config.use_binary)
        #     fv_perf = evaluate_four_variable(fv_valid_loader, model, dev, fp16=bool(config.fp16))
        #     total_perf = (performance + fv_perf) / 2
        #     if total_perf > best_performance:
        #         print(f"[Model Info] Saving the best model... with average performance {total_perf}..")
        #         best_performance = total_perf
        #         model_to_save = model.module if hasattr(model, "module") else model
        #         model_to_save.save_pretrained(f"model_files/{config.model_folder}")
        #         tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    print(f"[Model Info] Best validation performance: {best_performance}")
    model = UniversalModel.from_pretrained(f"model_files/{config.model_folder}").to(dev)
    if config.fp16:
        model.half()
        model.save_pretrained(f"model_files/{config.model_folder}")
        tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    return model


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
        dataset = UniversalDataset(file=conf.train_file, tokenizer=tokenizer, number=conf.train_num)
        print("[Data Info] Reading validation data", flush=True)
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, number=conf.dev_num)


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
        # evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16), use_binary=opt.use_binary)
    # else:
        # print(f"Testing the model now.")
        # model = ScoringModel.from_pretrained(f"model_files/{conf.model_folder}", num_labels=num_labels).to(conf.device)
        # print("[Data Info] Reading test data", flush=True)
        # eval_dataset = QuestionDataset(file=conf.dev_file, tokenizer=tokenizer, number=conf.dev_num,
        #                           use_binary=opt.use_binary, use_ans_string=opt.insert_m0_string)
        # fv_eval_dataset = FourVariableDataset(file=opt.fv_dev_file, tokenizer=tokenizer, number=conf.dev_num, insert_m0_string=opt.insert_m0_string)
        # print("[Data Info] Loading validation data", flush=True)
        # valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0,
        #                               collate_fn=eval_dataset.collate_function)
        # fv_valid_dataloader = DataLoader(fv_eval_dataset, batch_size=conf.batch_size, shuffle=False,
        #                                  num_workers=0, collate_fn=fv_eval_dataset.collate_function)
        # res_file= f"results/{conf.model_folder}.res.json"
        # err_file = f"results/{conf.model_folder}.err.json"
        # evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16), eval_answer=False,
        #          # result_file=res_file, err_file=err_file,
        #               idx2labels=idx2labels,
        #               pretty_idx2labels=pretty_idx2labels)
        # evaluate_four_variable(fv_valid_dataloader, model, conf.device, fp16=bool(conf.fp16))

if __name__ == "__main__":
    main()

