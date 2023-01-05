import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig, RobertaModel
from transformers.modeling_outputs import ModelOutput
from typing import (
    List, Optional, Tuple, Union, Any
)
from torch.nn import (
    CrossEntropyLoss,
    MSELoss,
    BCEWithLogitsLoss
)


def resize_position_embeddings(model, max_len):
    model_len = model.config.max_position_embeddings
    # position emb resize
    if isinstance(model, RobertaModel):
        max_len += 2
    pos_embed = nn.Embedding(max_len, model.config.hidden_size, padding_idx=model.config.pad_token_id)
    pos_embed.weight.data[:model_len] = model.embeddings.position_embeddings.weight.data
    pos_embed.weight.data[model_len:] = model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(max_len-model_len, 1)
    model.embeddings.position_embeddings = pos_embed
    model.embeddings.position_ids = torch.arange(max_len).expand((1, -1))
    model.embeddings.token_type_ids = torch.zeros(model.embeddings.position_ids.size(), dtype=torch.long)
    model.config.max_position_embeddings = max_len


@dataclass
class DetectionModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:, x.size(1)]
        return self.dropout(x)


class DetectionModel(nn.Module):

    def __init__(self, args, checkpoint=None):

        super(DetectionModel, self).__init__()
        self.num_labels = args.num_labels
        # self.label_balance_ratio = args.label_balance_ratio
        self.config = AutoConfig.from_pretrained(args.model)

        if args.mode == 'pair' or args.mode == 'seq':
            self.classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.config.hidden_dropout_prob),
                nn.Linear(self.config.hidden_size, self.num_labels)
            )
        else:
            self.pn_pos = PositionalEncoding(self.config.hidden_size)
            self.pn_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    batch_first=True
                ),
                num_layers=1
            )
            self.pn_init_emb = nn.Parameter(
                torch.empty([1, self.config.hidden_size]).normal_(mean=0.0, std=self.config.initializer_range)
            )
            self.pn_terminate_emb = nn.Parameter(
                torch.empty([1, self.config.hidden_size]).normal_(mean=0.0, std=self.config.initializer_range)
            )
            self.pn_qlinear = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.pn_klinear = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)

        self.encoder = AutoModel.from_pretrained(args.model)
        if args.max_source_length > self.encoder.config.max_position_embeddings:
            resize_position_embeddings(self.encoder, args.max_source_length)

        if checkpoint is not None:
            cp = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            self.load_state_dict(cp, strict=True)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def _classification(self, outputs, labels, bos_mask=None, label_pad_id=-100):

        hidden_states, pooler_out = outputs[0], outputs[1]
        if bos_mask is None:
            output = hidden_states[:, 0]
        else:
            output = hidden_states[bos_mask]

        logits = self.classifier(output)

        loss = None
        if labels is not None:
            label_mask = labels.ne(label_pad_id)
            cls_labels = labels[label_mask]
            assert cls_labels.size(0) == logits.size(0)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), cls_labels.squeeze())
                else:
                    loss = loss_fct(logits, cls_labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), cls_labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, cls_labels)
            # To enable parallel training with pooler output.
            loss += pooler_out.sum() * 0.

        return loss, logits

    def _pointer_net_training(self, outputs, labels, attention_mask, bos_mask, label_pad_id=-100):

        hidden_states, pooler_out = outputs[0], outputs[1]

        bsz = labels.size(0)

        # build padded cls hidden states
        cls_lengths = bos_mask.sum(-1).tolist()
        cls_list = torch.split(hidden_states[bos_mask], cls_lengths)
        cls = pad_sequence(cls_list, batch_first=True, padding_value=0.)

        # add termination state
        mem = torch.cat([self.pn_terminate_emb.unsqueeze(0).expand(bsz, 1, -1), cls], 1)
        mem_mask_list = [mem.new_zeros([length+1]) for length in cls_lengths]
        mem_mask = pad_sequence(mem_mask_list, batch_first=True, padding_value=-float('inf'))

        # build padded decoder input embs
        label_lengths = labels.ne(label_pad_id).sum(-1).tolist()
        pn_input_list = [mem[i][labels[i, :label_lengths[i]]+1] for i in range(bsz)]
        pn_input = pad_sequence(pn_input_list, batch_first=True, padding_value=0.)

        pn_input = torch.cat([self.pn_init_emb.unsqueeze(0).expand(bsz, 1, -1), pn_input], 1)
        pn_mask_list = [bos_mask.new_zeros([length+1]) for length in label_lengths]
        pn_mask = pad_sequence(pn_mask_list, batch_first=True, padding_value=1)

        # generate square mask
        tgt_len = pn_input.size(1)
        pn_square_mask = torch.triu(bos_mask.new_ones([tgt_len, tgt_len]), diagonal=1)

        # add positional embeddings
        pn_input = self.pn_pos(pn_input)

        # transformer decoder, pn_output = bsz * tgt * hid
        pn_output = self.pn_decoder(pn_input, hidden_states,
                                    tgt_mask=pn_square_mask,
                                    tgt_key_padding_mask=pn_mask,
                                    memory_key_padding_mask=~attention_mask.bool())

        # calculate score
        # mem: bsz * mem * hid
        # tgt: bsz * tgt * hid
        # logits: bsz * tgt * mem
        query = self.pn_qlinear(pn_output)
        key = self.pn_klinear(mem)
        logits = torch.matmul(query, key.transpose(1, 2)) + mem_mask.unsqueeze(1)

        # reconstruct labels
        label_list = [torch.cat([labels[i, :label_lengths[i]]+1, labels.new_zeros([1])], 0) for i in range(bsz)]
        cls_labels = pad_sequence(label_list, batch_first=True, padding_value=label_pad_id)

        # Avoid duplicate extraction.
        dup_mask = torch.zeros_like(logits)
        for i, la in enumerate(label_list):
            for j in range(1, la.size(0)):
                dup_mask[i, j] += dup_mask[i, j-1]
                if la[j-1] != 0:
                    dup_mask[i, j, la[j-1]] = -float('inf')
        logits += dup_mask

        # loss compute
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, mem.size(1)), cls_labels.view(-1))
        loss += pooler_out.sum() * 0.

        return loss, logits

    def _pointer_net_decoding(self, outputs, attention_mask, bos_mask):

        hidden_states = outputs[0]

        bsz = hidden_states.size(0)

        # build padded cls hidden states
        cls_lengths = bos_mask.sum(-1).tolist()
        cls_list = torch.split(hidden_states[bos_mask], cls_lengths)
        cls = pad_sequence(cls_list, batch_first=True, padding_value=0.)
        cls_mask_list = [bos_mask.new_ones([length]) for length in cls_lengths]
        cls_mask = pad_sequence(cls_mask_list, batch_first=True, padding_value=0)

        # add termination state
        mem = torch.cat([self.pn_terminate_emb.unsqueeze(0).expand(bsz, 1, -1), cls], 1)
        mem_mask_list = [mem.new_zeros([length+1]) for length in cls_lengths]
        mem_mask = pad_sequence(mem_mask_list, batch_first=True, padding_value=-float('inf'))

        # distribution size
        dist_size = mem.size(1)

        alive_seq = self.pn_init_emb.unsqueeze(0).expand(bsz, 1, -1)

        # dup_mask: avoid duplicate extraction
        dup_mask = hidden_states.new_zeros([bsz, dist_size])

        # record batch id
        batch_idx = torch.arange(bsz, device=bos_mask.device)

        # Structure that holds finished hypotheses.
        results = bos_mask.new_zeros([bsz, dist_size-1]).long()

        # count_down
        cnt_down = torch.full([bsz], self.num_labels-1, device=bos_mask.device)

        for _ in range(dist_size):

            # Decoder forward.
            # generate square mask
            tgt_len = alive_seq.size(1)
            pn_square_mask = torch.triu(bos_mask.new_ones([tgt_len, tgt_len]), diagonal=1)

            # add positional embeddings
            pn_input = self.pn_pos(alive_seq)

            # transformer decoder, pn_output = bsz * tgt * hid
            pn_output = self.pn_decoder(pn_input, hidden_states,
                                        tgt_mask=pn_square_mask,
                                        memory_key_padding_mask=~attention_mask.bool())

            # Generator forward.
            query = self.pn_qlinear(pn_output)
            key = self.pn_klinear(mem)
            logits = torch.matmul(query, key.transpose(1, 2))[:, -1]
            logits += mem_mask

            # Avoid duplicate extraction.
            logits += dup_mask
            probs = F.log_softmax(logits, -1)

            # Get score and index of the next step token.
            # sampling
            # m = Categorical(logits=probs)
            # ids = m.sample()
            # greedy search
            # probs[probs.exp()[:, 0] < 0.9, 0] = -1e20

            _, ids = probs.max(dim=-1)

            # update dup_mask
            dup_mask[F.one_hot(ids, dist_size).bool()] = -float('inf')
            dup_mask[:, 0] = 0.

            # Append last prediction.
            last_pre_hid = torch.cat([mem[i][ids[i]].unsqueeze(0) for i in range(ids.size(0))], 0)
            alive_seq = torch.cat([alive_seq, last_pre_hid.unsqueeze(1)], 1)

            # terminated if reach the stop state
            is_terminated = ids.eq(0)
            cnt_down[is_terminated] -= 1

            # If all examples are finished, no need to go further.
            non_finished = cnt_down.nonzero().view(-1)
            if len(non_finished) == 0:
                break

            # finished if the count down is in the end.
            is_finished = cnt_down.eq(0)

            # Save unfinished hypotheses.
            for i in range(is_finished.size(0)):
                if not is_terminated[i]:
                    results[batch_idx[i]][ids[i]-1] = cnt_down[i]

            # Remove finished batches for the next step.
            alive_seq = alive_seq.index_select(0, non_finished)
            batch_idx = batch_idx.index_select(0, non_finished)
            mem = mem.index_select(0, non_finished)
            mem_mask = mem_mask.index_select(0, non_finished)
            hidden_states = hidden_states.index_select(0, non_finished)
            attention_mask = attention_mask.index_select(0, non_finished)
            dup_mask = dup_mask.index_select(0, non_finished)
            cnt_down = cnt_down.index_select(0, non_finished)

        return F.one_hot(results[cls_mask], self.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        label_pad_id: Optional[int] = -100,
        mode: Optional[str] = 'pair',
        cls_token_id: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], DetectionModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]` or -100. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            Labels with indices set to `-100` are ignored.
        """

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if mode == 'seq' or mode == 'span':
            bos_mask = input_ids.eq(cls_token_id)
            # Make sure that the first position of cls set to False.
            # because it represent candidate summary rather than a dialogue sentence.
            bos_mask[:, 0] = False
        else:
            bos_mask = None

        # forward
        labels = labels.long()
        if mode == 'pair' or mode == 'seq':
            loss, logits = self._classification(outputs, labels, bos_mask, label_pad_id)
        else:
            if self.training:
                loss, logits = self._pointer_net_training(outputs, labels, attention_mask, bos_mask, label_pad_id)
            else:
                logits = self._pointer_net_decoding(outputs, attention_mask, bos_mask)
                loss = None
                if labels is not None:
                    # reconstruct labels
                    bsz = labels.size(0)
                    cls_lengths = bos_mask.sum(-1).tolist()
                    label_list = [labels.new_zeros([cls_lengths[i]]) for i in range(bsz)]
                    label_lengths = labels.ne(label_pad_id).sum(-1).tolist()
                    if self.num_labels > 2:
                        special_indexs = labels.eq(-1).nonzero().view(bsz, self.num_labels-2, -1).tolist()
                    for i in range(bsz):
                        if self.num_labels <= 2:
                            label_list[i][labels[i, :label_lengths[i]]] = 1
                        else:
                            for j in range(self.num_labels-1):
                                la = self.num_labels - j - 1
                                if j == 0:
                                    label_list[i][labels[i, :special_indexs[i][j][1]]] = la
                                if j == self.num_labels - 2:
                                    label_list[i][labels[i, special_indexs[i][j-1][1]+1:label_lengths[i]]] = la
                                else:
                                    label_list[i][labels[i, special_indexs[i][j-1][1]+1:special_indexs[i][j][1]]] = la
                    labels = pad_sequence(label_list, batch_first=True, padding_value=label_pad_id)

        return DetectionModelOutput(
            loss=loss,
            logits=logits,
            labels=labels,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
