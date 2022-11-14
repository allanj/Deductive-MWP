## Deductive Reasoning for Math Word Problem Solving 

This is the offcial repo for the ACL-2022 paper "[Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction](https://arxiv.org/abs/2203.10316)".

<img width="1341" alt="Screenshot 2022-11-14 at 4 42 03 PM" src="https://user-images.githubusercontent.com/3351187/201614535-b8323ca0-ed31-4bbc-947f-8e5f0b7b6f7d.png">

### Requirements
* transformers `pip3 install transformers`
* Pytorch > 1.7.1
* accelerate package `pip3 install accelerate` (for distributed training)


### Usage

Reproduce the results, simply run our scripts under `scripts` folder.

#### Math23k
For example, reproduce the results for `Math23k` dataset with train/val/test setting,
```shell
bash scripts/run_math23k.sh
```
Run the following for the train/test setting
```shell
bash scripts/run_math23k_train_test.sh
```

### Main Results
We reproduce the main results of **Roberta-base-DeductiveReasoner** in the following table.

| Dataset                  | Value Accuracy | 
|:-------------------------|:--------------:|
| Math23k (train/val/test) |      84.3      | 
| Math23k (train/test)     |      86.0      |
| MAWPS (5-fold CV)        |      92.0      | 
| MathQA (train/val/test)  |      78.6      |
| SVAMP                    |48.9|

More details can be found in Appendix C in our paper. 


### Checkpoints
We also provide the **Roberta-base-DeductiveReasoner** checkpoints that we have trained on the Math23k, MathQA and SVAMP datasets.
We do not provide the 5-fold model checkpoints due to space limitation.

|             Dataset              | Link  | 
|:--------------------------------:|---|
 | Math23k (train/dev/test setting) | [Link](https://drive.google.com/file/d/1TAHbdCKar0gqFzOd76LIYMQyI6hPOmL0/view?usp=sharing)  |
|   Math23k (train/test setting)   | [Link](https://huggingface.co/allanjie/math23k_train_test_roberta-base/tree/main)  |
 |              MathQA              | [Link](https://drive.google.com/file/d/1hgqSZwMyFearr_RJebL51ROflqwdsZUv/view?usp=sharing) | 
|              SVAMP               | [Link](https://drive.google.com/file/d/1ykI_pTPiCrHhgVA1gVN-yZeB-e0-J0TK/view?usp=sharing)  | 

### Datasets

The data for Math23k Five-fold is not uploaded to GitHub due to slightly larger dataset size, it is uploaded [here](https://drive.google.com/file/d/1oQZUPeIA6TlNySqjcZhTMQA4-onwljTU/view?usp=sharing) in Google Drive. 



### Citation
If you find this work useful, please cite our paper:
```
@inproceedings{jie2022learning,
  title={Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction},
  author={Jie, Zhanming and Li, Jierui and Lu, Wei},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={5944--5955},
  year={2022}
}
```
