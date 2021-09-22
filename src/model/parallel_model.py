from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from src.model.universal_model import UniversalOutput
from torch.nn.utils.rnn import pad_sequence



def get_combination_mask_from_variable_mask(variable_mask:torch.Tensor, combination: torch.Tensor):
    batch_size, num_variable = variable_mask.size()  ## batch_size x num_variable
    num_combinations, _ = combination.size()  ## 6
    temp = torch.gather(variable_mask, 1, combination.view(-1).unsqueeze(0).expand(batch_size, num_combinations * 2))
    batched_comb_mask = temp.unsqueeze(-1).view(batch_size, num_combinations, 2)
    return batched_comb_mask[:,:, 0] * batched_comb_mask[:,:, 1]


class ParallelModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig,
                 diff_param_for_height:bool=True,
                 height: int = 4,
                 constant_num: int = 0):
        """
        Constructor for model function
        :param config:
        :param diff_param_for_height: whether we want to use different layers/parameters for different height
        :param height: the maximum number of height we want to use
        :param constant_num: the number of constant we consider
        :param add_replacement: only at h=0, whether we want to consider somehting like "a*a" or "a+a"
                                also applies to h>0 when `consider_multplie_m0` = True
        :param consider_multiple_m0: considering more m0 in one single step. for example soemthing like "m3 = m1 x m2".
        """
        super().__init__(config)
        self.num_labels = config.num_labels ## should be 6
        assert self.num_labels == 6
        self.config = config

        self.bert = BertModel(config)

        self.label_rep2label = nn.Linear(config.hidden_size, 2) # 0 score and  1 score
        self.diff_param_for_height = diff_param_for_height
        self.max_height = height ## 3 operation
        self.linears = nn.ModuleList()
        if diff_param_for_height:
            for h in range(self.max_height):
                current_linears = nn.ModuleList()
                for i in range(self.num_labels):
                    current_linears.append(nn.Sequential(
                        nn.Linear(3 * config.hidden_size, config.hidden_size),
                        nn.ReLU(),
                        nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        nn.Dropout(config.hidden_dropout_prob)
                    ))
                self.linears.append(current_linears)
        else:
            for i in range(self.num_labels):
                self.linears.append(nn.Sequential(
                    nn.Linear(3 * config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                    nn.Dropout(config.hidden_dropout_prob)
                ))
        self.stopper_transformation = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                    nn.Dropout(config.hidden_dropout_prob)
                )

        self.stopper = nn.Linear(config.hidden_size, 2) ## whether we need to stop or not.
        self.variable_gru = None
        self.multihead_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=12, batch_first=True)
        self.constant_num = constant_num
        self.constant_emb = None
        if self.constant_num > 0:
            self.const_rep = nn.Parameter(torch.randn(self.constant_num, config.hidden_size))

        self.init_weights()


    def forward(self,
        input_ids=None, ## batch_size  x max_seq_length
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        variable_indexs_start: torch.Tensor = None, ## batch_size x num_variable
        variable_indexs_end: torch.Tensor = None,  ## batch_size x num_variable
        num_variables: torch.Tensor = None, # batch_size [3,4]
        variable_index_mask:torch.Tensor = None, # batch_size x num_variable
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ## binary label: (batch_size, max_height,  num_combinations, num_op_labels, 2).    (left_var_index, right_var_index, label_index, stop_label)
        label_height_mask = None, #  (batch_size, height)
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_eval=False
    ):
        r"""
                labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                    Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                    config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                    If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert( # batch_size, sent_len, hidden_size,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        batch_size, sent_len, hidden_size = outputs.last_hidden_state.size()
        if labels is not None and not is_eval:
            # is_train
            _, max_height, _, _, _ = labels.size()
        else:
            max_height = self.max_height

        _, max_num_variable = variable_indexs_start.size()

        var_sum = (variable_indexs_start - variable_indexs_end).sum() ## if add <NUM>, we can just choose one as hidden_states
        var_start_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_start.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
        if var_sum != 0:
            var_end_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_end.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
            var_hidden_states = var_start_hidden_states + var_end_hidden_states
        else:
            var_hidden_states= var_start_hidden_states
        if self.constant_num > 0:
            constant_hidden_states = self.const_rep.unsqueeze(0).expand(batch_size, self.constant_num, hidden_size)
            var_hidden_states = torch.cat([constant_hidden_states, var_hidden_states], dim=1)
            num_variables = num_variables + self.constant_num
            max_num_variable = max_num_variable + self.constant_num
            const_idx_mask = torch.ones((batch_size, self.constant_num), device=variable_indexs_start.device)
            variable_index_mask = torch.cat([const_idx_mask, variable_index_mask], dim = 1)

        best_mi_label_rep = None
        max_num_intermediate = None
        mi_mask = None
        loss = 0
        all_logits = []
        for i in range(max_height):
            linear_modules = self.linears[i] if self.diff_param_for_height else self.linears
            if i == 0:
                ## max_num_variable = 4. -> [0,1,2,3]
                num_var_range = torch.arange(0, max_num_variable, device=variable_indexs_start.device)
                ## 6x2 matrix
                combination = torch.combinations(num_var_range, r=2, with_replacement=True)  ##number_of_combinations x 2
                num_combinations, _ = combination.size()  # number_of_combinations x 2
                # batch_size x num_combinations. 2*6
                batched_combination_mask = get_combination_mask_from_variable_mask(variable_mask=variable_index_mask, combination=combination)  # batch_size, num_combinations

                var_comb_hidden_states = torch.gather(var_hidden_states, 1, combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size))
                # m0_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size * 3).sum(dim=-2)
                expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size)
                m0_hidden_states = torch.cat([expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :], expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]], dim=-1)
                # batch_size, num_combinations/num_m0, hidden_size: 2,6,768

                ## batch_size, num_combinations/num_m0, num_labels, hidden_size
                m0_label_rep = torch.stack([layer(m0_hidden_states) for layer in linear_modules], dim=2)
                ## batch_size, num_combinations/num_m0, num_labels, 2, 2
                m0_logits = self.label_rep2label(m0_label_rep).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2, 2)
                m0_logits = m0_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2, 2).log()
                ## batch_size, num_combinations/num_m0, num_labels, 2, 2
                m0_stopper_logits = self.stopper(self.stopper_transformation(m0_label_rep)).unsqueeze(-2).expand(batch_size, num_combinations, self.num_labels, 2, 2)

                ## batch_size, num_combinations/num_m0, num_labels, 2, 2
                m0_combined_logits = m0_logits + m0_stopper_logits

                all_logits.append(m0_combined_logits)
                if labels is not None and not is_eval:
                    mask_for_labels = batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2)
                    m0_gold_labels = labels[:, i, :num_combinations, :,  :]  ## (batch_size, num_combinations, num_op_labels, 2)
                    m0_gold_labels[mask_for_labels == 0] = -100
                    loss_fct = CrossEntropyLoss()
                    loss = loss + loss_fct(m0_combined_logits.view(-1, 2), m0_gold_labels.contiguous().view(-1))
                    mo_gold_label_tmp = m0_gold_labels.sum(dim=-1) ## (batch_size, num_combinations, num_op_labels)
                    judge = (mo_gold_label_tmp == 1).nonzero() ## non_zero_num x 3, -> (batch_idx, comb_idx, label_idx)
                else:
                    best_final_logits, best_final_label = m0_combined_logits.max(dim=-1)  ## batch_size, num_combinations/num_m0, num_labels, 2
                    best_final_label = torch.logical_or(best_final_label[:, :, :, 0], best_final_label[:, :, :, 1])  ## batch_size, num_combinations/num_m0, num_labels
                    judge = (best_final_label == 1).nonzero()
                num_in_batch = torch.bincount(judge[:,0], minlength=batch_size)  ## for example batch_size = 3 [2, 2, 4]..
                splits = torch.cumsum(num_in_batch, dim=0)[:-1].tolist() ## cumsum = [2,4,8] so the split point [2,4]
                max_num_intermediate = max(num_in_batch) ## max = 4
                if batch_size > 1:
                    list_of_tensors = torch.vsplit(judge, splits) ## tuple of tensors (2,3) (2,3) (4,3)
                    padded_judge = pad_sequence(list_of_tensors, batch_first=True).view(-1,3)
                else:
                    padded_judge = judge

                best_mi_label_rep = m0_label_rep[padded_judge[:, 0], padded_judge[:, 1], padded_judge[:, 2]]  ## batch_size x max_num_m0,  hidden_size
                best_mi_label_rep = best_mi_label_rep.view(batch_size, max_num_intermediate, hidden_size)

                # get mask
                temp_range = torch.arange(max_num_intermediate, device=m0_combined_logits.device).unsqueeze(0).expand(batch_size, max_num_intermediate) #[ [0,1,2,3], [0,1,2,3], [0,1,2,3]
                num_in_batch = num_in_batch.unsqueeze(1).expand(batch_size, max_num_intermediate) #[[2,2,2,2], [2,2,2,2], [4,4,4,4]]
                mi_mask = torch.lt(temp_range, num_in_batch) # batch_size x max_num_intermediate [[1,1,0,0], [1,1,0,0], [1,1,1,1]]
            else:
                ## update hidden_state (gated hidden state)
                var_attn_mask = torch.eye(max_num_intermediate + max_num_variable , max_num_intermediate + max_num_variable , device=best_mi_label_rep.device)

                expanded_mi_mask = mi_mask.unsqueeze(1).expand(batch_size, max_num_intermediate + max_num_variable, max_num_intermediate)
                expanded_var_attn_mask = var_attn_mask.unsqueeze(0).expand(batch_size, max_num_intermediate + max_num_variable, max_num_intermediate + max_num_variable).clone()
                expanded_var_attn_mask[:, :, :max_num_intermediate] = torch.logical_and(expanded_var_attn_mask[:, :, :max_num_intermediate], expanded_mi_mask)
                expanded_var_attn_mask = expanded_var_attn_mask.unsqueeze(1).expand(batch_size, self.multihead_attention.num_heads, max_num_intermediate + max_num_variable , max_num_intermediate + max_num_variable)
                expanded_var_attn_mask = 1 - (expanded_var_attn_mask.contiguous().view(-1, max_num_intermediate + max_num_variable , max_num_intermediate + max_num_variable))

                feature_out = torch.cat([best_mi_label_rep, var_hidden_states], dim=1)
                feature_out, _ = self.multihead_attention(query=feature_out, key=feature_out, value=feature_out, attn_mask=expanded_var_attn_mask)
                var_hidden_states = feature_out[:, max_num_intermediate:, :]

                num_var_range = torch.arange(0, max_num_variable + max_num_intermediate, device=variable_indexs_start.device)
                ## 6x2 matrix
                combination = torch.combinations(num_var_range, r=2,  with_replacement=True)  ##number_of_combinations x 2
                num_combinations, _ = combination.size()  # number_of_combinations x 2
                variable_index_mask = torch.cat([mi_mask, variable_index_mask], dim= 1)
                batched_combination_mask = get_combination_mask_from_variable_mask(variable_mask=variable_index_mask, combination=combination)

                var_hidden_states = torch.cat([best_mi_label_rep, var_hidden_states], dim=1)  ## batch_size x (num_var + i) x hidden_size
                var_comb_hidden_states = torch.gather(var_hidden_states, 1, combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size))
                expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size)
                mi_hidden_states = torch.cat( [expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],
                                        expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]], dim=-1)
                mi_label_rep = torch.stack([layer(mi_hidden_states) for layer in linear_modules], dim=2)
                mi_logits = self.label_rep2label(mi_label_rep).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2, 2)
                mi_logits = mi_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2, 2).log()

                mi_stopper_logits = self.stopper(self.stopper_transformation(mi_label_rep)).unsqueeze(-2).expand(batch_size, num_combinations, self.num_labels, 2, 2)
                mi_combined_logits = mi_logits + mi_stopper_logits
                all_logits.append(mi_combined_logits)


                max_num_variable += max_num_intermediate

                if labels is not None and not is_eval:
                    mask_for_labels = batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2)
                    mi_gold_labels = labels[:, i, :num_combinations, :,  :]   ## (batch_size, num_combinations, num_op_labels, 2)
                    mi_gold_labels[mask_for_labels == 0] = -100
                    loss_fct = CrossEntropyLoss()
                    loss = loss + loss_fct(mi_combined_logits.view(-1, 2), mi_gold_labels.contiguous().view(-1))
                    mi_gold_label_tmp = mi_gold_labels.sum(dim=-1)
                    judge = (mi_gold_label_tmp == 1).nonzero()
                else:
                    best_final_logits, best_final_label = mi_combined_logits.max(dim=-1)
                    best_final_label = torch.logical_or(best_final_label[:, :, :, 0], best_final_label[:, :, :, 1])
                    judge = (best_final_label == 1).nonzero()

                num_in_batch = torch.bincount(judge[:, 0], minlength=batch_size)  ## for example batch_size = 3 [2, 2, 4]..
                splits = torch.cumsum(num_in_batch, dim=0)[:-1].tolist()  ## cumsum = [2,4,8] so the split point [2,4]
                max_num_intermediate = max(num_in_batch)  ## max = 4
                if batch_size > 1:
                    list_of_tensors = torch.vsplit(judge, splits)  ## tuple of tensors (2,3) (2,3) (4,3)
                    assert len(list_of_tensors) == batch_size
                    padded_judge = pad_sequence(list_of_tensors, batch_first=True).view(-1, 3)
                else:
                    padded_judge = judge

                best_mi_label_rep = mi_label_rep[padded_judge[:, 0], padded_judge[:, 1], padded_judge[:,  2]]  ## batch_size x max_num_m0,  hidden_size
                best_mi_label_rep = best_mi_label_rep.view(batch_size, max_num_intermediate, hidden_size)
                # get mask
                temp_range = torch.arange(max_num_intermediate, device=mi_combined_logits.device).unsqueeze(0).expand(batch_size, max_num_intermediate)  # [ [0,1,2,3], [0,1,2,3], [0,1,2,3]
                num_in_batch = num_in_batch.unsqueeze(1).expand(batch_size, max_num_intermediate)
                mi_mask = torch.lt(temp_range, num_in_batch)


        return UniversalOutput(loss=loss, all_logits=all_logits)


def test_case_batch_one():
    model = ParallelModel.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=6, constant_num=2)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    text1 = "一本笔记本 <quant> 元钱, 王小明共带了 <quant> 元, 他一共能买多少本这样的笔记本?"  ## x= temp_b / temp_a
    text2 = "爸爸买来 <quant> 个桃子, 吃了 <quant> 个, 妈妈又买来 <quant> 个桃子, 现在有多少个桃子?"  ##x= temp_a - temp_b + temp_c"
    ## tokens = ['一', '本', '笔', '记', '本', '<', 'q', '##uan', '##t', '>', '元', '钱', ',', '王', '小', '明', '共', '带', '了', '<', 'q', '##uan', '##t', '>', '元', ',', '他', '一', '共', '能', '买', '多', '少', '本', '这', '样', '的', '笔', '记', '本', '?']
    res = tokenizer.batch_encode_plus([text1], return_tensors='pt', padding=True)
    input_ids = res["input_ids"]
    attention_mask = res["attention_mask"]
    token_type_ids = res["token_type_ids"]
    # variable_indexs_start = torch.tensor([[6, 20, 0], [5, 16, 28]])
    # variable_indexs_end = torch.tensor([[10, 24, 0], [9, 20, 32]])
    # num_variables = torch.tensor([2, 3])
    # variable_index_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    variable_indexs_start = torch.tensor([[6, 20]])
    variable_indexs_end = torch.tensor([[10, 24]])
    num_variables = torch.tensor([2])
    variable_index_mask = torch.tensor([[1, 1]])
    combs = torch.combinations(torch.arange(4), with_replacement=True)

    num_combination, _ = combs.size()
    ## batch_size = 2, height=2, 3
    labels = torch.zeros(1, 1, num_combination, 6, 2, dtype=torch.long)
    labels[0, 0, 8, 5, 1] = 1
    print(labels.size())
    print(model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                variable_indexs_start=variable_indexs_start,
                variable_indexs_end=variable_indexs_end,
                num_variables=num_variables,
                variable_index_mask=variable_index_mask,
                label_height_mask = None,
                labels=labels).loss)

def test_case_batch_two():
    model = ParallelModel.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=6, constant_num=2)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    text1 = "一本笔记本 <quant> 元钱, 王小明共带了 <quant> 元, 他一共能买多少本这样的笔记本?"  ## x= temp_b / temp_a
    text2 = "爸爸买来 <quant> 个桃子, 吃了 <quant> 个, 妈妈又买来 <quant> 个桃子, 现在有多少个桃子?"  ##x= temp_a - temp_b + temp_c "
    ## tokens = ['一', '本', '笔', '记', '本', '<', 'q', '##uan', '##t', '>', '元', '钱', ',', '王', '小', '明', '共', '带', '了', '<', 'q', '##uan', '##t', '>', '元', ',', '他', '一', '共', '能', '买', '多', '少', '本', '这', '样', '的', '笔', '记', '本', '?']
    res = tokenizer.batch_encode_plus([text1, text2], return_tensors='pt', padding=True)
    input_ids = res["input_ids"]
    attention_mask = res["attention_mask"]
    token_type_ids = res["token_type_ids"]
    variable_indexs_start = torch.tensor([[6, 20, 0], [5, 16, 28]])
    variable_indexs_end = torch.tensor([[10, 24, 0], [9, 20, 32]])
    num_variables = torch.tensor([2, 3])
    variable_index_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    combs = torch.combinations(torch.arange(7), with_replacement=True)

    num_combination, _ = combs.size()
    ## batch_size = 2, height=2, 3
    labels = torch.zeros(2, 2, num_combination, 6, 2, dtype=torch.long)
    labels[0, 0, 10, 5, 1] = 1 ## batch0: b/a
    labels[0, 0, 4] = -100 ## not involved combinations
    labels[0, 0, 8] = -100
    labels[0, 0, 11] = -100
    labels[0, 0, 13] = -100
    labels[0, 0, 14] = -100
    labels[0, 1] = -100  ## batch 0 has no height = 1

    labels[1, 0, 10, 1, 0] = 1 ## batch0, height = 0, a - b
    labels[1, 0, 11, 1, 0] = 1  ## batch0, height = 0, a - b
    labels[:, 0, 15:] = -100 ## for h=0, the combination only up to 15. torchcombintion(5)

    labels[1, 1, 5, 0, 1] = 1  ## batch0, height = 1, m0 + c
    print(labels.size())
    print(model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                variable_indexs_start=variable_indexs_start,
                variable_indexs_end=variable_indexs_end,
                num_variables=num_variables,
                variable_index_mask=variable_index_mask,
                label_height_mask=None,
                labels=labels).loss)

def test_case_batch_two_mutiple_m0():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = ParallelModel.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=6, constant_num=0, add_replacement=True, height=4, consider_multiple_m0=True)
    model.eval()
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    text1 = "一本笔记本 <quant> 元钱, 王小明共带了 <quant> 元, 他一共能买多少本这样的笔记本?"  ## x= temp_b / temp_a
    text2 = "爸爸买来 <quant> 个桃子, 吃了 <quant> 个, 妈妈又买来 <quant> 个桃子, 现在有多少个桃子?"  ##x= temp_a - temp_b + temp_c"
    ## tokens = ['一', '本', '笔', '记', '本', '<', 'q', '##uan', '##t', '>', '元', '钱', ',', '王', '小', '明', '共', '带', '了', '<', 'q', '##uan', '##t', '>', '元', ',', '他', '一', '共', '能', '买', '多', '少', '本', '这', '样', '的', '笔', '记', '本', '?']
    res = tokenizer.batch_encode_plus([text1, text2], return_tensors='pt', padding=True)
    input_ids = res["input_ids"]
    attention_mask = res["attention_mask"]
    token_type_ids = res["token_type_ids"]
    variable_indexs_start = torch.tensor([[6, 20, 0], [5, 16, 28]])
    variable_indexs_end = torch.tensor([[10, 24, 0], [9, 20, 32]])
    num_variables = torch.tensor([2, 3])
    variable_index_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    ## batch_size = 2, height=2, 3
    labels = torch.tensor([
        [
            [
                0, 1, uni_labels.index('/_rev'), 1
            ],
            [
                0, 0, 0, 0 ## 3 means, for this one, we directly forward
            ],
            [
                0, 0, 0, 0 ## 3 means, for this one, we directly forward
            ]
        ],
        [
            [
                0, 1, uni_labels.index('-'), 0
            ],
            [
                0, 3, uni_labels.index('+'), 1
            ],
            [
                0, 0, 0, 0 ## 3 means, for this one, we directly forward
            ]
        ]
    ])
    label_height_mask = torch.tensor(
        [
            [
                1, 0, 0
            ],
            [
                1, 1, 0
            ]
        ]
    )
    print(label_height_mask.size())
    print(labels.size())
    print(model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                variable_indexs_start=variable_indexs_start,
                variable_indexs_end=variable_indexs_end,
                num_variables=num_variables,
                variable_index_mask=variable_index_mask,
                label_height_mask = label_height_mask,
                labels=labels))


def set_seed():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

if __name__ == '__main__':
    set_seed()
    # test_case_batch_two_mutiple_m0()
    test_case_batch_two()