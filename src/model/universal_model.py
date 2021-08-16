from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

def get_combination_mask(batched_num_variables: torch.Tensor, combination: torch.Tensor):
    """

    :param batched_num_variables: (batch_size)
    :param combination: (num_combinations, 2) 6,2
    :return: batched_comb_mask: (batch_size, num_combinations)
    """
    batch_size, = batched_num_variables.size() ## [ 2,]
    num_combinations, _ = combination.size() ## 6
    batched_num_variables = batched_num_variables.expand(batch_size, num_combinations, 2) ## (2) -> (2,6,2)
    batched_combination = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)## (6, 2) -> (2,6,2)
    batched_comb_mask = torch.lt(batched_combination, batched_num_variables) ## batch_size, num_combinations, 2

    return batched_comb_mask[:,:, 0] * batched_comb_mask[:,:, 1]



class UniversalModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels ## should be 6
        self.config = config

        self.bert = BertModel(config)

        self.linears = nn.ModuleList()
        for i in range(self.num_labels):
            self.linears.append(nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                nn.Dropout(config.hidden_dropout_prob)
            ))

        self.label_rep2label = nn.Linear(config.hidden_size, 1) # 0 or 1
        self.max_height = 3 ## 3 operation
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
        ## (batch_size, height, 3). (left_var_index, right_var_index, label_index) when height>=1, left_var_index always -1, because left always m0
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
        if labels is not None:
            # is_train
            _, max_height, _ = labels.size()
        else:
            max_height = self.max_height

        batch_size, max_num_variable = variable_indexs_start.size()

        var_start_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_start.unsqueeze(-1).expand(batch_size, max_num_variable,  hidden_size))
        var_end_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_end.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
        var_hidden_states = var_start_hidden_states + var_end_hidden_states
        best_mi_label_rep = None
        loss = 0
        for i in range(max_height):
            if i == 0:
                ## max_num_variable = 4. -> [0,1,2,3]
                num_var_range = torch.arange(0, max_num_variable, device=variable_indexs_start.device)
                ## 6x2 matrix
                combination = torch.combinations(num_var_range, r=2, with_replacement=False)  ##number_of_combinations x 2
                num_combinations, _ = combination.size()  # number_of_combinations x 2
                # batch_size x num_combinations. 2*6
                batched_combination_mask = get_combination_mask(batched_num_variables=num_variables, combination=combination)  # batch_size, num_combinations

                var_comb_hidden_states = torch.gather(outputs.last_hidden_state, 1, combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size))
                m0_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size).sum(dim=-2)
                # batch_size, num_combinations/num_m0, hidden_size: 2,6,768

                ## batch_size, num_combinations/num_m0, num_labels, hidden_size
                m0_label_rep = torch.stack([layer(m0_hidden_states) for layer in self.linears], dim=2)
                ## batch_size, num_combinations/num_m0, num_labels
                m0_logits = self.label_rep2label(m0_label_rep).squeeze(-1)
                m0_logits = m0_logits + batched_combination_mask.unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels).log()

                best_temp_score, best_temp_label = m0_logits.max(dim=-1) ## batch_size, num_combinations
                best_m0_score, best_comb = best_temp_score.max(dim=-1) ## batch_size
                best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)## batch_size

                b_idxs = [k for k in range(batch_size)]
                best_m0_label_rep = m0_label_rep[b_idxs, best_comb, best_label] # batch_size x hidden_size
                best_mi_label_rep = best_m0_label_rep
                ## NOTE: add loosss
                if labels is not None:
                    m0_gold_labels = labels[:, i, :] ## batch_size x 3 (left_var_index, right_var_index, label_index)
                    m0_gold_comb = m0_gold_labels[:, :2].unsqueeze(1).expand(batch_size, num_combinations, 2)
                    batched_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                    judge = m0_gold_comb == batched_comb
                    judge = judge[:, :, 0] * judge[:, :, 1] #batch_size, num_combinations
                    judge = judge.non_zero()[:,0] #batch_size

                    m0_gold_scores = m0_logits[b_idxs, judge, m0_gold_labels[:, -1]] ## batch_size
                    loss += (best_m0_score - m0_gold_scores).sum()
            else:
                mi_sum_states = var_hidden_states + best_mi_label_rep.expand(batch_size, max_num_variable, hidden_size)
                ## batch_size, max_num_variable, num_labels, hidden_size
                mi_label_rep = torch.stack([layer(mi_sum_states) for layer in self.linears], dim=2)
                ## batch_size, max_num_variable, num_labels,
                mi_logits = self.label_rep2label(mi_label_rep).squeeze(-1)
                mi_logits = mi_logits + variable_index_mask.unsqueeze(-1).expand(batch_size, max_num_variable, self.num_labels).log()

                best_temp_score, best_temp_label = mi_logits.max(dim=-1)  ## batch_size, max_num_variable
                best_m0_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
                best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size

                b_idxs = [k for k in range(batch_size)]
                best_mi_label_rep = mi_label_rep[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                ## NOTE: add loss
                if labels is not None:
                    mi_gold_labels = labels[:, i, -2:]  ## batch_size x 2
                    mi_gold_scores = mi_logits[b_idxs, mi_gold_labels[:, 0], mi_gold_labels[:, 1]]  ## batch_size
                    loss += (best_m0_score - mi_gold_scores).sum()

        return loss
