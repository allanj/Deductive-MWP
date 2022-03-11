## Math Solving 

### Requirements
* transformers `pip3 install transformers`
* Pytorch > 1.7.1


### Usage

Reproduce the results, simply run our scripts under `scripts` folder.

For example, reproduce the results for `Math23k` dataset,
```shell
bash scripts/run_math23k.sh
```


### Checkpoints
We also provide checkpoints that we have trained on the Math23k dataset.

| Model  | Dataset  | Link  | 
|---|---|---|
| Roberta  | Math23k (train/dev/test)  | [Link]()  | 
| Roberta  | Math23k (train/test)  | [Link]()  | '
| Roberta  | Math23k (5-fold)  | [Link]()  | 
|  Roberta | MAWPS (5-fold) | [Link]()  | 
|  Roberta | MathQA  | [Link]() | 
|  Roberta |  SVAMP | [Link]()  | 


### Citation
If you find this work useful, please cite our paper:
```
@InProceedings{jie2022math, 
    author = "Jie, Zhanming and Li, Jierui and Lu, Wei", 
    title = "Math Word Problem Solving as Complex Relation Extraction", 
    booktitle = "Proceedings of ACL", 
    year = "2022"
}
```