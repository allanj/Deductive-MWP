from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import torch.nn as nn
import torch
import torch.utils.checkpoint
from src.model.universal_model import UniversalOutput, get_combination_mask

class UniversalModel_XLMRoberta(RobertaPreTrainedModel):

    def __init__(self, config: XLMRobertaConfig,
                 diff_param_for_height:bool=True,
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

        self.roberta = XLMRobertaModel(config)
        self.add_replacement = bool(add_replacement)
        self.consider_multiple_m0 = bool(consider_multiple_m0)

        self.label_rep2label = nn.Linear(config.hidden_size, 1) # 0 or 1
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
        outputs = self.roberta( # batch_size, sent_len, hidden_size,
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
            linear_modules = self.linears[i] if self.diff_param_for_height else self.linears
            if i == 0:
                ## max_num_variable = 4. -> [0,1,2,3]
                num_var_range = torch.arange(0, max_num_variable, device=variable_indexs_start.device)
                ## 6x2 matrix
                combination = torch.combinations(num_var_range, r=2,
                                                 with_replacement=self.add_replacement)  ##number_of_combinations x 2
                num_combinations, _ = combination.size()  # number_of_combinations x 2
                # batch_size x num_combinations. 2*6
                batched_combination_mask = get_combination_mask(batched_num_variables=num_variables,
                                                                combination=combination)  # batch_size, num_combinations

                var_comb_hidden_states = torch.gather(var_hidden_states, 1,
                                                      combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size,
                                                                                                             num_combinations * 2,
                                                                                                             hidden_size))
                # m0_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size * 3).sum(dim=-2)
                expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size,
                                                                                            num_combinations, 2,
                                                                                            hidden_size)
                m0_hidden_states = torch.cat(
                    [expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],
                     expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]], dim=-1)
                # batch_size, num_combinations/num_m0, hidden_size: 2,6,768

                ## batch_size, num_combinations/num_m0, num_labels, hidden_size
                m0_label_rep = torch.stack([layer(m0_hidden_states) for layer in linear_modules], dim=2)
                ## batch_size, num_combinations/num_m0, num_labels
                m0_logits = self.label_rep2label(m0_label_rep).expand(batch_size, num_combinations, self.num_labels, 2)
                m0_logits = m0_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size,
                                                                                                    num_combinations,
                                                                                                    self.num_labels,
                                                                                                    2).log()
                ## batch_size, num_combinations/num_m0, num_labels, 2
                m0_stopper_logits = self.stopper(self.stopper_transformation(m0_label_rep))

                var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)  ## batch_size x max_num_variable
                expanded_var_scores = torch.gather(var_scores, 1,
                                                   combination.unsqueeze(0).expand(batch_size, num_combinations,
                                                                                   2).view(batch_size, -1)).unsqueeze(
                    -1).view(batch_size, num_combinations, 2)
                expanded_var_scores = expanded_var_scores.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size,
                                                                                                         num_combinations,
                                                                                                         self.num_labels,
                                                                                                         2)

                ## batch_size, num_combinations/num_m0, num_labels, 2
                m0_combined_logits = m0_logits + m0_stopper_logits + expanded_var_scores

                all_logits.append(m0_combined_logits)
                best_temp_logits, best_stop_label = m0_combined_logits.max(
                    dim=-1)  ## batch_size, num_combinations/num_m0, num_labels
                best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)  ## batch_size, num_combinations
                best_m0_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
                best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size

                b_idxs = [k for k in range(batch_size)]
                # best_m0_label_rep = m0_label_rep[b_idxs, best_comb, best_label] # batch_size x hidden_size
                # best_mi_label_rep = best_m0_label_rep
                ## NOTE: add loosss
                if labels is not None and not is_eval:
                    m0_gold_labels = labels[:, i,
                                     :]  ## batch_size x 4 (left_var_index, right_var_index, label_index, stop_id)
                    m0_gold_comb = m0_gold_labels[:, :2].unsqueeze(1).expand(batch_size, num_combinations, 2)
                    batched_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                    judge = m0_gold_comb == batched_comb
                    judge = judge[:, :, 0] * judge[:, :, 1]  # batch_size, num_combinations
                    judge = judge.nonzero()[:, 1]  # batch_size

                    m0_gold_scores = m0_combined_logits[
                        b_idxs, judge, m0_gold_labels[:, 2], m0_gold_labels[:, 3]]  ## batch_size
                    loss = loss + (best_m0_score - m0_gold_scores).sum()

                    best_mi_label_rep = m0_label_rep[b_idxs, judge, m0_gold_labels[:, 2]]  ## teacher-forcing.
                    best_mi_scores = m0_logits[b_idxs, judge, m0_gold_labels[:, 2]][:, 0]  # batch_size
                else:
                    best_m0_label_rep = m0_label_rep[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                    best_mi_label_rep = best_m0_label_rep
                    best_mi_scores = m0_logits[b_idxs, best_comb, best_label][:, 0]  # batch_size
            else:
                if not self.consider_multiple_m0:
                    # best_mi_label_rep = self.intermediate_transformation(best_mi_label_rep)
                    # mi_sum_states = var_hidden_states + best_mi_label_rep.unsqueeze(1).expand(batch_size, max_num_variable, hidden_size)
                    expanded_best_mi_label_rep = best_mi_label_rep.unsqueeze(1).expand(batch_size, max_num_variable,
                                                                                       hidden_size)
                    mi_sum_states = torch.cat(
                        [expanded_best_mi_label_rep, var_hidden_states, expanded_best_mi_label_rep * var_hidden_states],
                        dim=-1)
                    ## batch_size, max_num_variable, num_labels, hidden_size
                    mi_label_rep = torch.stack([layer(mi_sum_states) for layer in linear_modules], dim=2)

                    ## batch_size, max_num_variable, num_labels,
                    mi_logits = self.label_rep2label(mi_label_rep).expand(batch_size, max_num_variable, self.num_labels,
                                                                          2)
                    mi_logits = mi_logits + variable_index_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size,
                                                                                                   max_num_variable,
                                                                                                   self.num_labels,
                                                                                                   2).log()

                    ## batch_size, max_num_variable, num_labels, 2
                    mi_stopper_logits = self.stopper(self.stopper_transformation(mi_label_rep))
                    ## batch_size, max_num_variable, num_labels, 2
                    mi_combined_logits = mi_logits + mi_stopper_logits

                    all_logits.append(mi_combined_logits)
                    best_temp_logits, best_stop_label = mi_combined_logits.max(
                        dim=-1)  ## batch_size, num_combinations/num_m0, num_labels
                    best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)  ## batch_size, max_num_variable
                    best_m0_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
                    best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size

                    b_idxs = [k for k in range(batch_size)]
                    # best_mi_label_rep = mi_label_rep[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                    ## NOTE: add loss
                    if labels is not None and not is_eval:
                        mi_gold_labels = labels[:, i, -3:]  ## batch_size x 3 (variable_index, label_id, stop_id)
                        height_mask = label_height_mask[:, i]  ## batch_size
                        mi_gold_scores = mi_combined_logits[
                            b_idxs, mi_gold_labels[:, 0], mi_gold_labels[:, 1], mi_gold_labels[:, 2]]  ## batch_size
                        current_loss = (
                                                   best_m0_score - mi_gold_scores) * height_mask  ## avoid compute loss for unnecessary height
                        loss = loss + current_loss.sum()
                        best_mi_label_rep = mi_label_rep[
                            b_idxs, mi_gold_labels[:, 0], mi_gold_labels[:, 1]]  ## teacher-forcing.
                    else:
                        best_mi_label_rep = mi_label_rep[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                else:
                    if self.var_update_mode == 0:
                        ## update hidden_state (gated hidden state)
                        init_h = best_mi_label_rep.unsqueeze(1).expand(batch_size, max_num_variable + i - 1,
                                                                       hidden_size).contiguous().view(-1, hidden_size)
                        gru_inputs = var_hidden_states.view(-1, hidden_size)
                        var_hidden_states = self.variable_gru(gru_inputs, init_h).view(batch_size,
                                                                                       max_num_variable + i - 1,
                                                                                       hidden_size)
                    elif self.var_update_mode == 1:
                        temp_states = torch.cat([best_mi_label_rep.unsqueeze(1), var_hidden_states],
                                                dim=1)  ## batch_size x (num_var + i) x hidden_size
                        temp_mask = torch.eye(max_num_variable + i, device=variable_indexs_start.device)
                        temp_mask[:, 0] = 1
                        temp_mask[0, :] = 1
                        updated_all_states, _ = self.variable_gru(temp_states, temp_states, temp_states,
                                                                  attn_mask=1 - temp_mask)
                        var_hidden_states = updated_all_states[:, 1:, :]

                    num_var_range = torch.arange(0, max_num_variable + i, device=variable_indexs_start.device)
                    ## 6x2 matrix
                    combination = torch.combinations(num_var_range, r=2,
                                                     with_replacement=self.add_replacement)  ##number_of_combinations x 2
                    num_combinations, _ = combination.size()  # number_of_combinations x 2
                    batched_combination_mask = get_combination_mask(batched_num_variables=num_variables + i,
                                                                    combination=combination)

                    var_hidden_states = torch.cat([best_mi_label_rep.unsqueeze(1), var_hidden_states],
                                                  dim=1)  ## batch_size x (num_var + i) x hidden_size
                    var_comb_hidden_states = torch.gather(var_hidden_states, 1,
                                                          combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(
                                                              batch_size, num_combinations * 2, hidden_size))
                    expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size,
                                                                                                num_combinations, 2,
                                                                                                hidden_size)
                    mi_hidden_states = torch.cat(
                        [expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],
                         expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]],
                        dim=-1)
                    mi_label_rep = torch.stack([layer(mi_hidden_states) for layer in linear_modules], dim=2)
                    mi_logits = self.label_rep2label(mi_label_rep).expand(batch_size, num_combinations, self.num_labels,
                                                                          2)
                    mi_logits = mi_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size,
                                                                                                        num_combinations,
                                                                                                        self.num_labels,
                                                                                                        2).log()

                    mi_stopper_logits = self.stopper(self.stopper_transformation(mi_label_rep))
                    var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)  ## batch_size x max_num_variable
                    expanded_var_scores = torch.gather(var_scores, 1,
                                                       combination.unsqueeze(0).expand(batch_size, num_combinations,
                                                                                       2).view(batch_size,
                                                                                               -1)).unsqueeze(-1).view(
                        batch_size, num_combinations, 2)
                    expanded_var_scores = expanded_var_scores.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size,
                                                                                                             num_combinations,
                                                                                                             self.num_labels,
                                                                                                             2)

                    mi_combined_logits = mi_logits + mi_stopper_logits + expanded_var_scores
                    all_logits.append(mi_combined_logits)
                    best_temp_logits, best_stop_label = mi_combined_logits.max(
                        dim=-1)  ## batch_size, num_combinations/num_m0, num_labels
                    best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)  ## batch_size, num_combinations
                    best_mi_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
                    best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size

                    ## NOTE: add loosss
                    if labels is not None and not is_eval:
                        mi_gold_labels = labels[:, i,
                                         :]  ## batch_size x 4 (left_var_index, right_var_index, label_index, stop_id)
                        mi_gold_comb = mi_gold_labels[:, :2].unsqueeze(1).expand(batch_size, num_combinations, 2)
                        batched_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                        judge = mi_gold_comb == batched_comb
                        judge = judge[:, :, 0] * judge[:, :, 1]  # batch_size, num_combinations
                        judge = judge.nonzero()[:, 1]  # batch_size

                        mi_gold_scores = mi_combined_logits[
                            b_idxs, judge, mi_gold_labels[:, 2], mi_gold_labels[:, 3]]  ## batch_size
                        height_mask = label_height_mask[:, i]  ## batch_size
                        current_loss = (
                                                   best_mi_score - mi_gold_scores) * height_mask  ## avoid compute loss for unnecessary height
                        loss = loss + current_loss.sum()
                        best_mi_label_rep = mi_label_rep[b_idxs, judge, mi_gold_labels[:, 2]]  ## teacher-forcing.
                        best_mi_scores = mi_logits[b_idxs, judge, mi_gold_labels[:, 2]][:, 0]  # batch_size
                    else:
                        best_mi_label_rep = mi_label_rep[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                        best_mi_scores = mi_logits[b_idxs, best_comb, best_label][:, 0]

        return UniversalOutput(loss=loss, all_logits=all_logits)


