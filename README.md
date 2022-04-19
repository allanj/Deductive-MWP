## Math Solving 

### Requirements
* transformers `pip3 install transformers`
* Pytorch > 1.7.1
* accelerate package `pip3 install accelerate` (for distributed training)


### Usage

Reproduce the results, simply run our scripts under `scripts` folder.

For example, reproduce the results for `Math23k` dataset,
```shell
bash scripts/run_math23k.sh
```

### Checkpoints
We also provide the **Roberta-base-DeductiveReasoner** checkpoints that we have trained on the Math23k, MathQA and SVAMP datasets.
We do not provide the 5-fold model checkpoints due to space limitation.

|             Dataset              | Link  | 
|:--------------------------------:|---|
 | Math23k (train/dev/test setting) | [Link](https://drive.google.com/file/d/1TAHbdCKar0gqFzOd76LIYMQyI6hPOmL0/view?usp=sharing)  | 
 |              MathQA              | [Link](https://drive.google.com/file/d/1hgqSZwMyFearr_RJebL51ROflqwdsZUv/view?usp=sharing) | 
|              SVAMP               | [Link](https://drive.google.com/file/d/1ykI_pTPiCrHhgVA1gVN-yZeB-e0-J0TK/view?usp=sharing)  | 

### Results


## TODO
- [ ] We plan to have a easier or more user-friendly way to load our pretrained model, by using the `from_pretrained` function from Hugginface. 

### Citation
If you find this work useful, please cite our paper:
```
@InProceedings{jie2022math, 
    author = "Jie, Zhanming and Li, Jierui and Lu, Wei", 
    title = "Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction", 
    booktitle = "Proceedings of ACL", 
    year = "2022"
}
```