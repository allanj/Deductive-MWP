from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    ModelOutput,
)
from dataclasses import dataclass
from typing import Optional, List

from src.model.beam_search_scorer import BeamSearchScorer

@dataclass
class UniversalOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    all_logits: List[torch.FloatTensor] = None

def get_combination_mask(batched_num_variables: torch.Tensor, combination: torch.Tensor):
    """

    :param batched_num_variables: (batch_size)
    :param combination: (num_combinations, 2) 6,2
    :return: batched_comb_mask: (batch_size, num_combinations)
    """
    batch_size, = batched_num_variables.size() ## [ 2,]
    num_combinations, _ = combination.size() ## 6
    batched_num_variables = batched_num_variables.unsqueeze(1).unsqueeze(2).expand(batch_size, num_combinations, 2) ## (2) -> (2,6,2)
    batched_combination = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)## (6, 2) -> (2,6,2)
    batched_comb_mask = torch.lt(batched_combination, batched_num_variables) ## batch_size, num_combinations, 2

    return batched_comb_mask[:,:, 0] * batched_comb_mask[:,:, 1]


class UniversalModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig,
                 height: int = 4,
                 constant_num: int = 0,
                 add_replacement: bool = False,
                 consider_multiple_m0: bool = False, var_update_mode: str= 'gru'):
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
        assert self.num_labels == 6 or self.num_labels == 8
        self.config = config

        self.bert = BertModel(config)
        self.add_replacement = bool(add_replacement)
        self.consider_multiple_m0 = bool(consider_multiple_m0)

        self.label_rep2label = nn.Linear(config.hidden_size, 1) # 0 or 1
        self.max_height = height ## 3 operation
        self.linears = nn.ModuleList()
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
        if var_update_mode == 'gru':
            self.var_update_mode = 0
        elif var_update_mode == 'attn':
            self.var_update_mode = 1
        else:
            self.var_update_mode = -1
        if self.consider_multiple_m0:
            if var_update_mode == 'gru':
                self.variable_gru = nn.GRUCell(config.hidden_size, config.hidden_size)
            elif var_update_mode == 'attn':
                self.variable_gru = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=6, batch_first=True)
            else:
                print("[WARNING] no rationalizer????????")
                self.variable_gru = None
        self.constant_num = constant_num
        self.constant_emb = None
        if self.constant_num > 0:
            self.const_rep = nn.Parameter(torch.randn(self.constant_num, config.hidden_size))
            # self.multihead_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=6, batch_first=True)

        self.variable_scorer = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                    nn.Dropout(config.hidden_dropout_prob),
                    nn.Linear(config.hidden_size, 1),
                )

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
        ## (batch_size, height, 4). (left_var_index, right_var_index, label_index, stop_label) when height>=1, left_var_index always -1, because left always m0
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
            _, max_height, _ = labels.size()
        else:
            max_height = self.max_height

        _, max_num_variable = variable_indexs_start.size()

        var_sum = (variable_indexs_start - variable_indexs_end).sum() ## if add <NUM>, we can just choose one as hidden_states
        var_start_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_start.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
        if var_sum != 0:
            var_end_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_end.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
            var_hidden_states = var_start_hidden_states + var_end_hidden_states
        else:
            var_hidden_states = var_start_hidden_states
        if self.constant_num > 0:
            constant_hidden_states = self.const_rep.unsqueeze(0).expand(batch_size, self.constant_num, hidden_size)
            var_hidden_states = torch.cat([constant_hidden_states, var_hidden_states], dim=1)
            num_variables = num_variables + self.constant_num
            max_num_variable = max_num_variable + self.constant_num
            const_idx_mask = torch.ones((batch_size, self.constant_num), device=variable_indexs_start.device)
            variable_index_mask = torch.cat([const_idx_mask, variable_index_mask], dim = 1)

            # updated_all_states, _ = self.multihead_attention(var_hidden_states, var_hidden_states, var_hidden_states,key_padding_mask=variable_index_mask)
            # var_hidden_states = torch.cat([updated_all_states[:, :2, :], var_hidden_states[:, 2:, :]], dim=1)

        best_mi_label_rep = None
        loss = 0
        all_logits = []
        best_mi_scores = None
        for i in range(max_height):
            linear_modules = self.linears
            if i == 0:
                ## max_num_variable = 4. -> [0,1,2,3]
                num_var_range = torch.arange(0, max_num_variable, device=variable_indexs_start.device)
                ## 6x2 matrix
                combination = torch.combinations(num_var_range, r=2, with_replacement=self.add_replacement)  ##number_of_combinations x 2
                num_combinations, _ = combination.size()  # number_of_combinations x 2
                # batch_size x num_combinations. 2*6
                batched_combination_mask = get_combination_mask(batched_num_variables=num_variables, combination=combination)  # batch_size, num_combinations

                var_comb_hidden_states = torch.gather(var_hidden_states, 1, combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size))
                # m0_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size * 3).sum(dim=-2)
                expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size)
                m0_hidden_states = torch.cat([expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :], expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]], dim=-1)
                # batch_size, num_combinations/num_m0, hidden_size: 2,6,768

                ## batch_size, num_combinations/num_m0, num_labels, hidden_size
                m0_label_rep = torch.stack([layer(m0_hidden_states) for layer in linear_modules], dim=2)
                ## batch_size, num_combinations/num_m0, num_labels
                m0_logits = self.label_rep2label(m0_label_rep).expand(batch_size, num_combinations, self.num_labels, 2)
                m0_logits = m0_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2).float().log()
                ## batch_size, num_combinations/num_m0, num_labels, 2
                m0_stopper_logits = self.stopper(self.stopper_transformation(m0_label_rep))

                var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)  ## batch_size x max_num_variable
                expanded_var_scores = torch.gather(var_scores, 1, combination.unsqueeze(0).expand(batch_size, num_combinations, 2).view(batch_size, -1)).unsqueeze(-1).view(batch_size, num_combinations, 2)
                expanded_var_scores = expanded_var_scores.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2)

                ## batch_size, num_combinations/num_m0, num_labels, 2
                m0_combined_logits = m0_logits + m0_stopper_logits + expanded_var_scores

                all_logits.append(m0_combined_logits)
                best_temp_logits, best_stop_label =  m0_combined_logits.max(dim=-1) ## batch_size, num_combinations/num_m0, num_labels
                best_temp_score, best_temp_label = best_temp_logits.max(dim=-1) ## batch_size, num_combinations
                best_m0_score, best_comb = best_temp_score.max(dim=-1) ## batch_size
                best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)## batch_size

                b_idxs = [k for k in range(batch_size)]
                # best_m0_label_rep = m0_label_rep[b_idxs, best_comb, best_label] # batch_size x hidden_size
                # best_mi_label_rep = best_m0_label_rep
                ## NOTE: add loosss
                if labels is not None and not is_eval:
                    m0_gold_labels = labels[:, i, :] ## batch_size x 4 (left_var_index, right_var_index, label_index, stop_id)
                    m0_gold_comb = m0_gold_labels[:, :2].unsqueeze(1).expand(batch_size, num_combinations, 2)
                    batched_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                    judge = m0_gold_comb == batched_comb
                    judge = judge[:, :, 0] * judge[:, :, 1] #batch_size, num_combinations
                    judge = judge.nonzero()[:,1] #batch_size

                    m0_gold_scores = m0_combined_logits[b_idxs, judge, m0_gold_labels[:, 2], m0_gold_labels[:, 3]] ## batch_size
                    loss = loss + (best_m0_score - m0_gold_scores).sum()

                    best_mi_label_rep = m0_label_rep[b_idxs, judge, m0_gold_labels[:, 2]] ## teacher-forcing.
                    best_mi_scores = m0_logits[b_idxs, judge, m0_gold_labels[:, 2]][:, 0]  # batch_size
                else:
                    best_m0_label_rep = m0_label_rep[b_idxs, best_comb, best_label] # batch_size x hidden_size
                    best_mi_label_rep = best_m0_label_rep
                    best_mi_scores = m0_logits[b_idxs, best_comb, best_label][:, 0]  # batch_size
            else:
                if not self.consider_multiple_m0:
                    # best_mi_label_rep = self.intermediate_transformation(best_mi_label_rep)
                    # mi_sum_states = var_hidden_states + best_mi_label_rep.unsqueeze(1).expand(batch_size, max_num_variable, hidden_size)
                    expanded_best_mi_label_rep = best_mi_label_rep.unsqueeze(1).expand(batch_size, max_num_variable, hidden_size)
                    mi_sum_states = torch.cat([expanded_best_mi_label_rep, var_hidden_states, expanded_best_mi_label_rep * var_hidden_states], dim= -1)
                    ## batch_size, max_num_variable, num_labels, hidden_size
                    mi_label_rep = torch.stack([layer(mi_sum_states) for layer in linear_modules], dim=2)

                    ## batch_size, max_num_variable, num_labels,
                    mi_logits = self.label_rep2label(mi_label_rep).expand(batch_size, max_num_variable, self.num_labels, 2)
                    mi_logits = mi_logits + variable_index_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, max_num_variable, self.num_labels, 2).float().log()

                    ## batch_size, max_num_variable, num_labels, 2
                    mi_stopper_logits = self.stopper(self.stopper_transformation(mi_label_rep))
                    ## batch_size, max_num_variable, num_labels, 2
                    mi_combined_logits = mi_logits + mi_stopper_logits

                    all_logits.append(mi_combined_logits)
                    best_temp_logits, best_stop_label = mi_combined_logits.max( dim=-1)  ## batch_size, num_combinations/num_m0, num_labels
                    best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)  ## batch_size, max_num_variable
                    best_m0_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
                    best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size

                    b_idxs = [k for k in range(batch_size)]
                    # best_mi_label_rep = mi_label_rep[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                    ## NOTE: add loss
                    if labels is not None and not is_eval:
                        mi_gold_labels = labels[:, i, -3:]  ## batch_size x 3 (variable_index, label_id, stop_id)
                        height_mask = label_height_mask[:, i] ## batch_size
                        mi_gold_scores = mi_combined_logits[b_idxs, mi_gold_labels[:, 0], mi_gold_labels[:, 1], mi_gold_labels[:, 2]]  ## batch_size
                        current_loss = (best_m0_score - mi_gold_scores) * height_mask ## avoid compute loss for unnecessary height
                        loss = loss + current_loss.sum()
                        best_mi_label_rep = mi_label_rep[b_idxs, mi_gold_labels[:, 0], mi_gold_labels[:, 1]]  ## teacher-forcing.
                    else:
                        best_mi_label_rep = mi_label_rep[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                else:
                    if self.var_update_mode == 0:
                        ## update hidden_state (gated hidden state)
                        init_h = best_mi_label_rep.unsqueeze(1).expand(batch_size, max_num_variable + i - 1, hidden_size).contiguous().view(-1, hidden_size)
                        gru_inputs = var_hidden_states.view(-1, hidden_size)
                        var_hidden_states = self.variable_gru(gru_inputs, init_h).view(batch_size, max_num_variable + i - 1, hidden_size)
                    elif self.var_update_mode == 1:
                        temp_states = torch.cat([best_mi_label_rep.unsqueeze(1), var_hidden_states], dim=1)  ## batch_size x (num_var + i) x hidden_size
                        temp_mask = torch.eye(max_num_variable + i, device=variable_indexs_start.device)
                        temp_mask[:, 0] = 1
                        temp_mask[0, :] = 1
                        updated_all_states, _ = self.variable_gru(temp_states, temp_states, temp_states, attn_mask=1 - temp_mask)
                        var_hidden_states = updated_all_states[:, 1:, :]

                    num_var_range = torch.arange(0, max_num_variable + i, device=variable_indexs_start.device)
                    ## 6x2 matrix
                    combination = torch.combinations(num_var_range, r=2,  with_replacement=self.add_replacement)  ##number_of_combinations x 2
                    num_combinations, _ = combination.size()  # number_of_combinations x 2
                    batched_combination_mask = get_combination_mask(batched_num_variables=num_variables + i, combination=combination)

                    var_hidden_states = torch.cat([best_mi_label_rep.unsqueeze(1), var_hidden_states], dim=1)  ## batch_size x (num_var + i) x hidden_size
                    var_comb_hidden_states = torch.gather(var_hidden_states, 1, combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size))
                    expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size)
                    mi_hidden_states = torch.cat( [expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],
                                            expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]], dim=-1)
                    mi_label_rep = torch.stack([layer(mi_hidden_states) for layer in linear_modules], dim=2)
                    mi_logits = self.label_rep2label(mi_label_rep).expand(batch_size, num_combinations, self.num_labels, 2)
                    mi_logits = mi_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2).float().log()

                    mi_stopper_logits = self.stopper(self.stopper_transformation(mi_label_rep))
                    var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)  ## batch_size x max_num_variable
                    expanded_var_scores = torch.gather(var_scores, 1, combination.unsqueeze(0).expand(batch_size, num_combinations, 2).view(batch_size,-1)).unsqueeze(-1).view(batch_size, num_combinations, 2)
                    expanded_var_scores = expanded_var_scores.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels,  2)

                    mi_combined_logits = mi_logits + mi_stopper_logits + expanded_var_scores
                    all_logits.append(mi_combined_logits)
                    best_temp_logits, best_stop_label = mi_combined_logits.max( dim=-1)  ## batch_size, num_combinations/num_m0, num_labels
                    best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)  ## batch_size, num_combinations
                    best_mi_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
                    best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size

                    ## NOTE: add loosss
                    if labels is not None and not is_eval:
                        mi_gold_labels = labels[:, i, :]  ## batch_size x 4 (left_var_index, right_var_index, label_index, stop_id)
                        mi_gold_comb = mi_gold_labels[:, :2].unsqueeze(1).expand(batch_size, num_combinations, 2)
                        batched_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                        judge = mi_gold_comb == batched_comb
                        judge = judge[:, :, 0] * judge[:, :, 1]  # batch_size, num_combinations
                        judge = judge.nonzero()[:, 1]  # batch_size

                        mi_gold_scores = mi_combined_logits[b_idxs, judge, mi_gold_labels[:, 2], mi_gold_labels[:, 3]]  ## batch_size
                        height_mask = label_height_mask[:, i]  ## batch_size
                        current_loss = (best_mi_score - mi_gold_scores) * height_mask  ## avoid compute loss for unnecessary height
                        loss = loss + current_loss.sum()
                        best_mi_label_rep = mi_label_rep[b_idxs, judge, mi_gold_labels[:, 2]]  ## teacher-forcing.
                        best_mi_scores = mi_logits[b_idxs, judge, mi_gold_labels[:, 2]][:, 0]  # batch_size
                    else:
                        best_mi_label_rep = mi_label_rep[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                        best_mi_scores = mi_logits[b_idxs, best_comb, best_label][:, 0]

        return UniversalOutput(loss=loss, all_logits=all_logits)

    def beam_search(self,
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
        ## (batch_size, height, 4). (left_var_index, right_var_index, label_index, stop_label) when height>=1, left_var_index always -1, because left always m0
        label_height_mask = None, #  (batch_size, height)
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_eval=False,
        num_beams:int = 1
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
            _, max_height, _ = labels.size()
        else:
            max_height = self.max_height

        _, max_num_variable = variable_indexs_start.size()

        var_sum = (variable_indexs_start - variable_indexs_end).sum() ## if add <NUM>, we can just choose one as hidden_states
        var_start_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_start.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
        if var_sum != 0:
            var_end_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_end.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
            var_hidden_states = var_start_hidden_states + var_end_hidden_states
        else:
            var_hidden_states = var_start_hidden_states
        if self.constant_num > 0:
            constant_hidden_states = self.const_rep.unsqueeze(0).expand(batch_size, self.constant_num, hidden_size)
            var_hidden_states = torch.cat([constant_hidden_states, var_hidden_states], dim=1)
            num_variables = num_variables + self.constant_num
            max_num_variable = max_num_variable + self.constant_num
            const_idx_mask = torch.ones((batch_size, self.constant_num), device=variable_indexs_start.device)
            variable_index_mask = torch.cat([const_idx_mask, variable_index_mask], dim = 1)

            # updated_all_states, _ = self.multihead_attention(var_hidden_states, var_hidden_states, var_hidden_states,key_padding_mask=variable_index_mask)
            # var_hidden_states = torch.cat([updated_all_states[:, :2, :], var_hidden_states[:, 2:, :]], dim=1)

        best_mi_label_rep = None
        all_logits = []
        b_idxs = torch.tensor([k for k in range(batch_size)]).long()

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=var_hidden_states.device,
            length_penalty=1
        )

        # beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        # beam_scores[:, :] = -1e9
        # beam_scores = beam_scores.view((batch_size * num_beams,))



        for i in range(max_height):
            linear_modules = self.linears[i] if self.diff_param_for_height else self.linears
            if i == 0:
                ## max_num_variable = 4. -> [0,1,2,3]
                num_var_range = torch.arange(0, max_num_variable, device=variable_indexs_start.device)
                ## 6x2 matrix
                combination = torch.combinations(num_var_range, r=2, with_replacement=self.add_replacement)  ##number_of_combinations x 2
                num_combinations, _ = combination.size()  # number_of_combinations x 2
                # batch_size x num_combinations. 2*6
                batched_combination_mask = get_combination_mask(batched_num_variables=num_variables, combination=combination)  # batch_size, num_combinations

                var_comb_hidden_states = torch.gather(var_hidden_states, 1, combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size))
                # m0_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size * 3).sum(dim=-2)
                expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size,  num_combinations, 2, hidden_size)
                m0_hidden_states = torch.cat(
                    [expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],
                     expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]], dim=-1)
                # batch_size, num_combinations/num_m0, hidden_size: 2,6,768

                ## batch_size, num_combinations/num_m0, num_labels, hidden_size
                m0_label_rep = torch.stack([layer(m0_hidden_states) for layer in linear_modules], dim=2)
                ## batch_size, num_combinations/num_m0, num_labels
                m0_logits = self.label_rep2label(m0_label_rep).expand(batch_size, num_combinations, self.num_labels, 2)
                m0_logits = m0_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2).log()
                ## batch_size, num_combinations/num_m0, num_labels, 2
                m0_stopper_logits = self.stopper(self.stopper_transformation(m0_label_rep))

                var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)  ## batch_size x max_num_variable
                expanded_var_scores = torch.gather(var_scores, 1, combination.unsqueeze(0).expand(batch_size, num_combinations, 2).view(batch_size, -1)).unsqueeze(-1).view(batch_size, num_combinations, 2)
                expanded_var_scores = expanded_var_scores.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2)

                ## batch_size, num_combinations/num_m0, num_labels, 2
                m0_combined_logits = m0_logits + m0_stopper_logits + expanded_var_scores

                all_logits.append(m0_combined_logits)
                # best_temp_logits, best_stop_label =  m0_combined_logits.max(dim=-1) ## batch_size, num_combinations/num_m0, num_labels
                # best_temp_score, best_temp_label = best_temp_logits.max(dim=-1) ## batch_size, num_combinations
                # best_m0_score, best_comb = best_temp_score.max(dim=-1) ## batch_size
                # best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)## batch_size



                m0_combined_scores = nn.functional.log_softmax(
                    m0_combined_logits.view(batch_size, -1), dim=-1
                )
                top_k_values, top_k_index = m0_combined_scores.view(batch_size, -1).topk(k=2 * num_beams, dim=-1, largest=True, sorted=True) ## batch_size x 2beam_size
                ## Note: we want to get.
                ## Note: current_best_beam: (batch_size, beam_size, 4 ](left_idx, right_idx, label_id, stop_id)

                best_previous_beam_idx = torch.floor(top_k_index / (num_combinations * self.num_labels * 2)).long()
                temp = top_k_index - num_combinations * self.num_labels * 2 * best_previous_beam_idx
                best_a = torch.floor(temp/(self.num_labels * 2)).long()  ## comb_id. (batch_size, beamsize*2)
                best_b = torch.floor((temp - self.num_labels * 2 * best_a)/2).long() ## label_id. (batch_size, beamsize*2)
                best_c = torch.remainder(top_k_index, 2).long() #2: stop id, (batch_size, beamsize*2)

                assert (best_previous_beam_idx >= 0).all() and (best_previous_beam_idx <= num_beams).all()
                assert (best_a >= 0).all() and (best_a <= num_combinations).all()
                assert (best_b >= 0).all() and (best_b <= self.num_labels).all()
                assert (best_c >= 0).all() and (best_c <= 2).all()

                batch_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                best_comb = batch_comb[b_idxs.repeat_interleave(2 * num_beams), best_a.view(-1)].view(batch_size, 2 * num_beams, 2)

                # current_best_beam: (batch_size, 2 * beam_size, 4 )
                next_labels = torch.cat([best_comb, best_b.unsqueeze(-1), best_c.unsqueeze(-1)], dim=-1).long()

                pad_previous_indices = torch.zeros((batch_size, 2 * num_beams), dtype=torch.long, device=next_labels.device).long()
                res = beam_scorer.process(current_best_beam=None,
                                          next_scores=top_k_values,
                                          next_labels=next_labels,
                                          next_comb_idx=best_a,
                                          best_previous_beam_indices = pad_previous_indices)
                next_beam_scores= res["next_beam_scores"]
                next_beam_labels = res["next_beam_labels"]
                next_best_previous_beam_indices =  res["next_best_previous_beam_indices"]
                next_beam_comb_idx = res["next_beam_comb_idx"]

                var_hidden_states = var_hidden_states.repeat_interleave(num_beams, dim=0)

                ## batch_size x beam_size, hidden size
                best_mi_label_rep = m0_label_rep[b_idxs.repeat_interleave(num_beams), next_beam_comb_idx.view(-1), next_beam_labels[:, :, 2].view(-1)]
                best_mi_beam_scores = next_beam_scores

                previous_beam_scores = next_beam_scores ## batch_size, num_beams
                previous_beam_comb_idx = next_beam_comb_idx
                current_beam_labels = next_beam_labels.unsqueeze(-2)  ## batch_size, self.num_beams, 1, 4
                initial_comb_num = num_combinations
            else:


                ## update hidden_state (gated hidden state)
                init_h = best_mi_label_rep.unsqueeze(1).expand(batch_size * num_beams, max_num_variable + i - 1, hidden_size).contiguous().view(-1, hidden_size)
                gru_inputs = var_hidden_states.view(-1, hidden_size)
                var_hidden_states = self.variable_gru(gru_inputs, init_h).view(batch_size * num_beams, max_num_variable + i - 1, hidden_size)

                num_var_range = torch.arange(0, max_num_variable + i, device=variable_indexs_start.device)
                ## 6x2 matrix
                combination = torch.combinations(num_var_range, r=2, with_replacement=self.add_replacement)  ##number_of_combinations x 2
                num_combinations, _ = combination.size()  # number_of_combinations x 2
                batched_combination_mask = get_combination_mask(batched_num_variables=num_variables + i, combination=combination)
                batched_combination_mask = batched_combination_mask.repeat_interleave(num_beams, dim=0)

                var_hidden_states = torch.cat([best_mi_label_rep.unsqueeze(1), var_hidden_states], dim=1)  ## batch_size x (num_var + i) x hidden_size
                var_comb_hidden_states = torch.gather(var_hidden_states, 1, combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size  * num_beams, num_combinations * 2, hidden_size))
                expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size * num_beams, num_combinations, 2, hidden_size)
                mi_hidden_states = torch.cat(
                    [expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],
                     expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]], dim=-1)
                mi_label_rep = torch.stack([layer(mi_hidden_states) for layer in linear_modules], dim=2)
                mi_logits = self.label_rep2label(mi_label_rep).expand(batch_size * num_beams, num_combinations, self.num_labels, 2)
                mi_logits = mi_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size * num_beams, num_combinations, self.num_labels, 2).log()

                mi_stopper_logits = self.stopper(self.stopper_transformation(mi_label_rep))
                var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)  ## batch_size x max_num_variable
                expanded_var_scores = torch.gather(var_scores, 1, combination.unsqueeze(0).expand(batch_size * num_beams, num_combinations, 2).view(batch_size * num_beams, -1)).unsqueeze(-1).view(batch_size * num_beams, num_combinations, 2)
                expanded_var_scores = expanded_var_scores.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size * num_beams, num_combinations, self.num_labels, 2)

                ## batch_size * num_beams, num_combinations/num_m0, num_labels, 2
                mi_combined_logits = mi_logits + mi_stopper_logits + expanded_var_scores
                all_logits.append(mi_combined_logits)
                # best_temp_logits, best_stop_label = mi_combined_logits.max(dim=-1)  ## batch_size, num_combinations/num_m0, num_labels
                # best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)  ## batch_size, num_combinations
                # best_mi_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
                # best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size

                ## (batch_size, num_beams, top_initial_comb_num * num_labels * 2)
                de_number = initial_comb_num * self.num_labels *2
                mi_combined_logits_top_comb, mi_combined_logits_top_comb_idx = mi_combined_logits.view(batch_size, num_beams, -1).topk(k=de_number,  dim=2, largest=True, sorted=True)
                offset = torch.arange(num_beams, device=mi_combined_logits.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_beams, de_number) * (num_combinations * self.num_labels *2)
                mi_combined_logits_top_comb_idx = mi_combined_logits_top_comb_idx + offset
                mi_combined_logits = mi_combined_logits_top_comb

                # batch_size, num_beams, (num_combinations/num_m0 *  num_labels * 2)
                mi_combined_scores = nn.functional.log_softmax(
                    mi_combined_logits.view(batch_size, num_beams, -1), dim=-1
                )
                # mi_combined_scores = mi_combined_scores + torch.log(torch.ones(mi_combined_scores.size(), device=mi_combined_scores.device) * 1.2)
                transformed_previous_beam_scores = previous_beam_scores.unsqueeze(-1).expand_as(mi_combined_scores.view(batch_size, num_beams, -1))#.contiguous().view(batch_size, -1)
                mi_beam_added_score = mi_combined_scores + transformed_previous_beam_scores
                top_k_values, top_k_index = mi_beam_added_score.view(batch_size, -1).topk(k=2 * num_beams, dim=-1)  ## batch_size x 2*beam_size

                top_k_index = torch.gather(mi_combined_logits_top_comb_idx.view(batch_size, -1), 1, top_k_index)

                best_previous_beam_idx = torch.floor(top_k_index / (num_combinations * self.num_labels * 2)).long()
                temp = top_k_index - num_combinations * self.num_labels * 2 * best_previous_beam_idx
                best_a = torch.floor(temp / (self.num_labels * 2)).long()  ## comb_id. (batch_size, beamsize*2)
                best_b = torch.floor(
                    (temp - self.num_labels * 2 * best_a) / 2).long()  ## label_id. (batch_size, beamsize*2)
                best_c = torch.remainder(top_k_index, 2).long()  # 2: stop id, (batch_size, beamsize*2)
                assert (best_previous_beam_idx >= 0).all() and (best_previous_beam_idx <= num_beams).all()
                assert (best_a >= 0).all() and (best_a <= num_combinations).all()
                assert (best_b >= 0).all() and (best_b <= self.num_labels).all()
                assert (best_c >= 0).all() and (best_c <= 2).all()



                batch_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                best_comb = batch_comb[b_idxs.repeat_interleave(2 * num_beams), best_a.view(-1)].view(batch_size, 2 * num_beams, 2)

                # (batch_size, 2 * beam_size, 4)
                next_labels = torch.cat([best_comb, best_b.unsqueeze(-1), best_c.unsqueeze(-1)], dim=-1).long()

                # best_previous_beam_labels = torch.gather(current_beam_labels, 1,
                #                                          best_previous_beam_idx.unsqueeze(-1).unsqueeze(-1).
                #                                          expand(batch_size, 2*num_beams, i, 4)).long()
                res = beam_scorer.process(current_best_beam=current_beam_labels,
                                          next_scores=top_k_values,
                                          next_labels=next_labels,
                                          next_comb_idx=best_a,
                                          best_previous_beam_indices=best_previous_beam_idx)

                next_beam_scores = res["next_beam_scores"]
                next_beam_labels = res["next_beam_labels"]
                next_best_previous_beam_indices = res["next_best_previous_beam_indices"]
                next_beam_comb_idx = res["next_beam_comb_idx"]

                var_hidden_states = torch.gather(var_hidden_states.view(batch_size, num_beams, max_num_variable + i, hidden_size), 1,
                                                 next_best_previous_beam_indices.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_beams, max_num_variable + i, hidden_size))

                best_mi_label_rep = mi_label_rep[
                    b_idxs.repeat_interleave(num_beams), next_beam_comb_idx.view(-1), next_beam_labels[:, :, 2].view(-1)]

                previous_beam_scores = next_beam_scores  ## batch_size, num_beams
                target_current_beam_labels = torch.gather(current_beam_labels, 1, next_best_previous_beam_indices.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_beams, i, 4))
                current_beam_labels = torch.cat([target_current_beam_labels, next_beam_labels.unsqueeze(-2)], dim=-2)
                prev_comb_num = num_combinations
                if beam_scorer.is_done:
                    break

        final_res = beam_scorer.finalize(
            current_best_beam=current_beam_labels,
            final_beam_scores=previous_beam_scores,
            max_height=max_height,
        )


        return final_res["decoded"], final_res["best_scores"]



def test_case_batch_two():
    model = UniversalModel.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=6, constant_num=2)
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
                0, 1, uni_labels.index('/_rev'), 0
            ],
            [
                -1, 2, 1, 1 ## 3 means, for this one, we directly forward
            ],
            [
                -1, 0, 0, 0 ## 3 means, for this one, we directly forward
            ]
        ],
        [
            [
                0, 1, uni_labels.index('-'), 0
            ],
            [
                -1, 2, uni_labels.index('+'), 0
            ],
            [
                -1, 3, 0, 1 ## 3 means, for this one, we directly forward
            ]
        ]
    ])
    label_height_mask = torch.tensor(
        [
            [
                1, 1, 0
            ],
            [
                1, 1, 1
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

def test_case_batch_two_mutiple_m0():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = UniversalModel.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=6, constant_num=0, add_replacement=True, height=4, consider_multiple_m0=True)
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

def test_beam_search():
    import random
    import numpy as np
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    model = UniversalModel.from_pretrained('model_files/ours_fp16_best', num_labels=6, constant_num=13, diff_param_for_height = False,
                 height= 10,
                 add_replacement= True,
                 consider_multiple_m0 = True)
    model.eval()
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    text1 = "一本笔记本 <quant> 元钱, 王小明共带了 <quant> 元, 他一共能买多少本这样的笔记本?"  ## x= temp_b / temp_a
    text2 = "爸爸买来 <quant> 个桃子, 吃了 <quant> 个, 妈妈又买来 <quant> 个桃子, 现在有多少个桃子?"  ##x= temp_a - temp_b + temp_c"
    res = tokenizer.batch_encode_plus([text1, text2], return_tensors='pt', padding=True)
    input_ids = res["input_ids"]
    attention_mask = res["attention_mask"]
    token_type_ids = res["token_type_ids"]
    variable_indexs_start = torch.tensor([[6, 20, 0], [5, 16, 28]])
    variable_indexs_end = torch.tensor([[10, 24, 0], [9, 20, 32]])
    num_variables = torch.tensor([2, 3])
    variable_index_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    res = model.beam_search(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                variable_indexs_start=variable_indexs_start,
                variable_indexs_end=variable_indexs_end,
                num_variables=num_variables,
                variable_index_mask=variable_index_mask,
                num_beams=3)
    print(res[0])
    print(res[1])

    from universal_main import get_batched_prediction_consider_multiple_m0
    from src.data.universal_dataset import UniFeature
    res = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                variable_indexs_start=variable_indexs_start,
                variable_indexs_end=variable_indexs_end,
                num_variables=num_variables,
                variable_index_mask=variable_index_mask)
    feature = UniFeature(variable_indexs_start=variable_indexs_start,input_ids=input_ids,
                         attention_mask=attention_mask)
    batched_prediction = get_batched_prediction_consider_multiple_m0(feature=feature, all_logits=res.all_logits,
                                                constant_num=13,
                                                add_replacement=True)
    ## post process remve extra
    for b, inst_predictions in enumerate(batched_prediction):
        for p, prediction_step in enumerate(inst_predictions):
            left, right, op_id, stop_id = prediction_step
            if stop_id == 1:
                batched_prediction[b] = batched_prediction[b][:(p + 1)]
                break
    print(batched_prediction)
if __name__ == '__main__':
    test_beam_search()