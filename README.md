## Math Solving 

### Requirements
* transformers `pip3 install transformers`
* Pytorch
* sacrebleu `pip3 install sacrebleu`  (this one is used if we work on generation)

### Data Preparation
You can **ignore** this part if you use the data directly from the server.
#### Four variable data preparation
Given the dataset `four_var_cases.json`
1. Generate the m0 description. (Run generation model by
    ```shell
    nohup bash run_seq2seq.bash > log 2>&1 &
    ```
    __Remember__ to change the file names in `preprocess/process_four_variables.py` before 
    run the above scirpts.
   
2. The first step gives us the data file `data/all_generated_1.0.json`. 
   Split the file into training and validation set. Run `python3 count.py`.
   This gives us `data/fv_train.json` and `data/fv_test.json`

### Usage

Before running, commands to execute
```shell
mkdir logs # logs folder to save the log
```


#### Multi-Task Model
Run the multi-task model (i.e., model for three-variable dataset and model for four-variable dataset).
```shell
nohup bash run_mtl.bash > log 2>&1 &
```
The corresponding main file is `mtl_main.py`

#### Old Model for Three-variable Dataset (Deprecated)
```shell
nohup bash run_ours.bash > log 2>&1 &
```
The corresponding main file is `main.py`


#### Generation Model (Generate the m0 intermediate description)
```shell
nohup bash run_seq2seq.bash > log 2>&1 &
```
The corresponding main file is `gen_main.py`.