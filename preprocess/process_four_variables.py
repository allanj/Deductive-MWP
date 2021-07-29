import sys
import torch
from src.utils import read_data,write_data
from tqdm import tqdm
from transformers import MBartTokenizerFast, MBartForConditionalGeneration

var_name2idx = {"v1": 0, "v2": 1, "v3": 2}

combinations = [
    "v1 v2",
    "v1 v3",
    "v2 v3"
]

operations = [
    "+", "-", "-_rev", "x", "/", "/_rev"
]

def get_stat(file: str):
    data =read_data(file= file)
    total = len(data)
    print(f"total number of data: {total}")

def get_concat_string(numerical_vars, left_idx, right_idx, operation):
    left_text = numerical_vars[left_idx]["concat_text"]
    right_text = numerical_vars[right_idx]["concat_text"]
    res = None
    if operation == "+":
        res = left_text + " + " + right_text
    elif operation == "-":
        res = left_text + " - " + right_text
    elif operation == "-_rev":
        res = right_text + " - " + left_text
    elif operation == "x":
        res = left_text + " * " + right_text
    elif operation == "/":
        res = left_text + " / " + right_text
    elif operation == "/_rev":
        res = right_text + " / " + left_text
    else:
        raise NotImplementedError
    return res

def generate_description(model:MBartForConditionalGeneration, tokenizer: MBartTokenizerFast, data, repetition_penalty: float):
    res = []
    for key in tqdm(data, desc="generating data", total=len(data)):
        value = data[key]
        id = key
        variables = value["variables"]
        numerical_vars = [{}] * 3
        ans_var = None
        for order, varibale in enumerate(variables):
            var_name, number, tokens = varibale
            concat_text = ""
            for token in tokens:
                if token.startswith("<") and token.endswith(">"):
                    concat_text += " <quant> "
                else:
                    concat_text += token
            updated_obj = {
                "var_name": var_name,
                "number": number,
                "tokens": tokens,
                "order_in_text": order,
                "concat_text": concat_text
            }
            if var_name == "x":
                ans_var = updated_obj
                continue
            numerical_vars[var_name2idx[var_name]] = updated_obj
        input_texts = []
        complete_generations = []
        for comb in combinations:
            left_idx, right_idx = comb.split(" ")
            left_idx = int(left_idx[1:]) - 1
            right_idx = int(right_idx[1:]) - 1
            for operation in operations:
                combined_text = get_concat_string(numerical_vars, left_idx, right_idx, operation)
                input_texts.append(combined_text)
                complete_generations.append({
                    "comb": comb, "operator": operation
                })
        assert  len(input_texts) == len(complete_generations)

        tokenized_res = tokenizer.batch_encode_plus(input_texts, add_special_tokens=True, padding=True, return_tensors='pt')
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            generated_ids = model.generate(
                input_ids=tokenized_res["input_ids"].to(device),
                attention_mask=tokenized_res["attention_mask"].to(device),
                max_length=100,
                num_beams=1,
                use_cache=True,
                repetition_penalty=repetition_penalty
            )

            preds = [g.strip() for g in
                     tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
            for i, prediction in enumerate(preds):
                complete_generations[i]["generated_m0"] = prediction
        res.append({
            "id": id,
            "full_text": value["full_text"],
            "variables": numerical_vars,
            "ans_variable": ans_var,
            "all_generated_m0": complete_generations,
            "equation": value["equation"],
            "context": value["context"],
            "m0": value["m0"]
        })
    return res


def load_generation_model(model_folder:str, device:torch.device):
    tokenizer = MBartTokenizerFast.from_pretrained(f"model_files/{model_folder}")
    print("[Model Info] Loading the saved model", flush=True)
    model = MBartForConditionalGeneration.from_pretrained(f"model_files/{model_folder}").to(device)
    return model, tokenizer

if __name__ == '__main__':
    device = torch.device('cuda:0')
    data = read_data(file="data/four_var_cases_updated.json")
    repetition_penalty = float(sys.argv[1]) #1.1
    output_file = f"data/all_generated_{repetition_penalty}_updated.json"
    model_folder = sys.argv[2]#"generation"
    model, tokenizer = load_generation_model(model_folder=model_folder, device=device)
    res = generate_description(model=model, tokenizer=tokenizer, data = data, repetition_penalty = repetition_penalty)
    write_data(data=res, file=output_file)
