from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

class ScoringModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels ## should be 6
        self.config = config

        self.bert = BertModel(config)

        self.linears = nn.ModuleList()
        for i in range(self.num_labels):
            self.linears.append(nn.Sequential(
                nn.Linear(2 * config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                nn.Dropout(config.hidden_dropout_prob)
            ))

        self.label_rep2label = nn.Linear(config.hidden_size, 2) # 0 or 1
        self.init_weights()

    def forward(self,
        dataset=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        sent_starts: torch.Tensor = None, sent_ends: torch.Tensor = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if dataset[0] == "3_var":
            return self.forward_three_variable(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                sent_starts=sent_starts,
                                sent_ends=sent_ends,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                labels=labels,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict)
        elif dataset[0] == "4_var":
            return self.forward_four_variable(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                sent_starts=sent_starts,
                                sent_ends=sent_ends,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                labels=labels,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict)
        else:
            raise NotImplementedError(f"forward not implemented for {dataset}")

    def forward_three_variable(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        sent_starts: torch.Tensor = None, sent_ends: torch.Tensor = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
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


        outputs = self.bert(
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


        last_hidden_state = outputs.last_hidden_state
        batch_size, _, hidden_size = last_hidden_state.size()
        sent_start_states = torch.gather(last_hidden_state, 1, sent_starts.unsqueeze(2).expand(batch_size, -1, hidden_size))
        sent_end_states = torch.gather(last_hidden_state, 1, sent_ends.unsqueeze(2).expand(batch_size, -1, hidden_size))
        ## batch_size, num_variables, hidden_size
        sent_states = torch.cat([sent_start_states, sent_end_states], dim=-1)
        ## batch_size, hidden_size
        summed_states = sent_states.sum(dim=-2)
        ## batch_size, num_labels, hidden_size
        label_rep = torch.stack([layer(summed_states) for layer in self.linears], dim = 1)

        ## batch_size, num_labels, 2 (0,1)
        logits = self.label_rep2label(label_rep)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_four_variable(self,
        input_ids=None, ## batch_size x num_m0 x max_seq_length
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        sent_starts: torch.Tensor = None, ## batch_size x num_m0 x 3
        sent_ends: torch.Tensor = None,  ## batch_size x num_m0 x 3
        head_mask=None,
        inputs_embeds=None,
        labels=None, ## batch_size x num_m0
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

        batch_size, num_m0, max_seq_length = input_ids.size()

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
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



        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = last_hidden_state.view(batch_size, num_m0, last_hidden_state.size(-2), last_hidden_state.size(-1))

        batch_size, _, _, hidden_size = last_hidden_state.size()
        sent_start_states = torch.gather(last_hidden_state, 2, sent_starts.unsqueeze(3).expand(batch_size, num_m0, -1, hidden_size))
        sent_end_states = torch.gather(last_hidden_state, 2, sent_ends.unsqueeze(3).expand(batch_size, num_m0, -1, hidden_size))

        # ## batch_size, num_m0, num_variables, hidden_size
        sent_states = torch.cat([sent_start_states, sent_end_states], dim=-1)
        ## batch_size, num_m0, hidden_size
        summed_states = sent_states.sum(dim=-2)

        ## batch_size, num_m0, num_labels, hidden_size
        label_rep = torch.stack([layer(summed_states) for layer in self.linears], dim=2)

        ## batch_size, num_m0, num_labels, 2 (0,1)
        logits = self.label_rep2label(label_rep)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
