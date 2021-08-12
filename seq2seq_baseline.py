from src.data.seq2seq_dataset import Seq2SeqDataset
from src.config import Config
from torch.utils.data import DataLoader
from transformers import MBartTokenizerFast, PreTrainedTokenizer
from tqdm import tqdm
from transformers import MBartForConditionalGeneration
import argparse
from src.utils import get_optimizers, read_data, write_data
import torch
import torch.nn as nn
import numpy as np
import os
import random
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from sacrebleu import corpus_bleu, sentence_bleu

def calculate_bleu_score(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": sentence_bleu(output_lns, refs_lns, **kwargs).score}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed_all(args.seed)


def parse_arguments(parser: argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--batch_size', type=int, default=50, help="default batch size is 10 (works well)")
    parser.add_argument('--max_seq_length', type=int, default=300, help="maximum sequence length")
    parser.add_argument('--generated_max_length', type=int, default=100, help="maximum target length")
    parser.add_argument('--train_num', type=int, default=-1, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=-1, help="The number of development data, -1 means all data")

    parser.add_argument('--train_file', type=str, default="data/complex/train.json")
    parser.add_argument('--dev_file', type=str, default="data/complex/validation.json")

    parser.add_argument('--seed', type=int, default=42, help="random seed")

    # model
    parser.add_argument('--model_folder', type=str, default="seq2seq",
                        help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="facebook",
                        help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="mbart-large-cc25", help="The bert model name to used")

    # training
    parser.add_argument('--mode', type=str, default="train", help="training or testing")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate of the AdamW optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")
    parser.add_argument('--num_epochs', type=int, default=30, help="The number of epochs to run")

    parser.add_argument('--fp16', type=int, default=0, choices=[0,1], help="fp16")

    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args



def train(config: Config, train_dataloader: DataLoader, num_epochs: int,
          bert_model_name: str, dev: torch.device, valid_dataloader: DataLoader = None,
          tokenizer: PreTrainedTokenizer = None):
    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * num_epochs)

    model = MBartForConditionalGeneration.from_pretrained(bert_model_name)
    model.to(dev)
    if config.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config.fp16))
    optimizer, scheduler = get_optimizers(config, model, t_total, warmup_step=0, eps=1e-8, weight_decay=0.0)
    optimizer.zero_grad()
    model.zero_grad()
    best_accuracy = -1
    os.makedirs(f"model_files/{config.model_folder}", exist_ok=True)  ## create model files. not raise error if exist
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for iter, batch in tqdm(enumerate(train_dataloader, 1), desc="--training batch", total=len(train_dataloader)):
            target_id = batch.labels.to(dev)
            lm_labels = target_id.clone()
            lm_labels[target_id == tokenizer.pad_token_id] = -100
            input_ids = batch.input_ids.to(dev)
            mask = batch.attention_mask.to(dev)
            decoder_input_ids = shift_tokens_right(target_id, tokenizer.pad_token_id)
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                outputs = model(input_ids, attention_mask=mask, decoder_input_ids=decoder_input_ids, labels=lm_labels)
            loss = outputs[0]
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
            optimizer.zero_grad()
            scheduler.step()
            model.zero_grad()
            if iter % 1000 == 0:
                print(f"epoch: {epoch}, iteration: {iter}, current mean loss: {total_loss / iter:.2f}", flush=True)
        print(f"Finish epoch: {epoch}, loss: {total_loss:.2f}, mean loss: {total_loss / len(train_dataloader):.2f}",
              flush=True)
        if valid_dataloader is not None:
            model.eval()
            accuracy = test(config=config, valid_dataloader=valid_dataloader, model=model, dev=dev, tokenizer=tokenizer)
            if accuracy > best_accuracy:
                print(f"[Model Info] Saving the best model...")
                tokenizer.save_pretrained(f"model_files/{config.model_folder}")
                model.save_pretrained(f"model_files/{config.model_folder}")
                best_accuracy = accuracy
    print(f"[Model Info] Returning the best model")
    model = MBartForConditionalGeneration.from_pretrained(f"model_files/{config.model_folder}").to(dev)
    if config.fp16:
        model.half()
        model.save_pretrained(f"model_files/{config.model_folder}")
        tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    return model

def test(config: Config, valid_dataloader: DataLoader, model: nn.Module, dev: torch.device,
         tokenizer: PreTrainedTokenizer, result_file:str=None):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for index, batch in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            generated_ids = model.generate(
                input_ids=batch.input_ids.to(dev),
                attention_mask=batch.attention_mask.to(dev),
                max_length=config.generated_max_length,
                num_beams=1,
                use_cache=True
            )

            preds = [g.strip() for g in tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
            predictions.extend(preds)
            target = [g.strip() for g in tokenizer.batch_decode(batch.labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
            targets.extend(target)

    print("####PREDICTIIONS####")
    print(predictions[:20])
    print("####TARGETS####")
    print(targets[:20])

    correct = 0
    average_bleu = 0
    for pred, gold in zip(predictions, targets):
        if pred == gold:
            correct += 1
            average_bleu += 1
        else:
            try:
                bleu_score = calculate_bleu_score(pred, [gold])['bleu']
            except Exception as e:
                bleu_score = 0
            average_bleu += bleu_score
    average_bleu = average_bleu / len(predictions)
    accuracy = (correct * 1.0 / len(predictions)) * 100
    print(f"[Info] Acc.:{(correct * 1.0 / len(predictions)) * 100:.2f}, Total: {len(predictions)}, BLEU: {average_bleu}", flush=True)
    if result_file is not None:
        res = []
        for feature, prediction, gold in zip(valid_dataloader.dataset._features, predictions, targets):
            input = tokenizer.decode(feature.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            res.append({
                "input": input,
                "gold": gold,
                "prediction": prediction
            })
        write_data(file=result_file, data=res)

    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Cloze Test question answering")
    opt = parse_arguments(parser)
    set_seed(opt)
    conf = Config(opt)

    bert_model_name = f'{conf.bert_folder}/{conf.bert_model_name}'
    tokenizer = MBartTokenizerFast.from_pretrained(bert_model_name)

    # Read dataset
    print("[Data Info] Reading training data", flush=True)
    dataset = Seq2SeqDataset(tokenizer=tokenizer, file=conf.train_file, number=conf.train_num)
    print("[Data Info]  Reading validation data", flush=True)
    eval_dataset = Seq2SeqDataset(tokenizer=tokenizer, file=conf.dev_file, number=conf.dev_num)

    # Prepare data loader
    if opt.mode == "train":
        print("[Data Info] Loading training data", flush=True)
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True,
                                      num_workers=conf.num_workers,
                                      collate_fn=dataset.collate_function)
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False,
                                      num_workers=conf.num_workers,
                                      collate_fn=eval_dataset.collate_function)

        # Train the model
        model = train(conf, train_dataloader,
                      num_epochs=conf.num_epochs,
                      bert_model_name=bert_model_name,
                      valid_dataloader=valid_dataloader,
                      dev=conf.device,
                      tokenizer=tokenizer)
        test(conf, valid_dataloader, model, conf.device, tokenizer)
    elif opt.mode == "test":
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False,
                                      num_workers=conf.num_workers,
                                      collate_fn=eval_dataset.collate_function)
        print("[Model Info] Loading the saved model", flush=True)
        model = MBartForConditionalGeneration.from_pretrained(f"model_files/{conf.model_folder}").to(conf.device)
        test(conf, valid_dataloader, model, conf.device, tokenizer, result_file=f"results/{conf.model_folder}_res.json")


if __name__ == "__main__":
    main()
