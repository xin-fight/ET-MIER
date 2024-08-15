import math
import torch
import pickle
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
from long_seq import process_long_input
from losses_SAIS_Evidence import ATLoss, SAISLoss, MLLTRSCLloss, Relation_Specificloss
from constant_SAIS_Evidence import *
from Model_assistant_SAIS_Evidence import *
from prepro import docred_ent2id


class DocREModel(nn.Module):
    def __init__(self, config, model, tokenizer,
                 emb_size=768, block_size=64, num_labels=-1,
                 max_sent_num=25, evi_thresh=0.2):
        """
        Initialize the model.
        :model: Pretrained langage model encoder;
        :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
        :emb_size: Dimension of embeddings for subject/object (head/tail) representations;
        :block_size: Number of blocks for grouped bilinear classification;
        :num_labels: Maximum number of relation labels for each entity pair;
        :max_sent_num: Maximum number of sentences for each document;
        :evi_thresh: Threshold for selecting evidence sentences.
        """
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)

        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh

        self.three_atten = three_atten

        self.use_cls_info = use_cls_info
       
        self.dropout = nn.Dropout(0.1)
        self.reduced_dim = reduced_dim
        self.relation_dim = relation_dim
       
        self.use_bilinear = use_bilinear
       
        self.Relation_Specific_num = Relation_Specific_num
       
        self.Rel_Spe_loss = Relation_Specificloss()
       
        self.SCL_loss = MLLTRSCLloss(tau=tau, tau_base=tau_base)
        self.lambda_3 = lambda_3

        self.extractor_dim = extractor_dim

        if not self.use_cls_info:
           
            self.head_extractor = nn.Linear(2 * self.reduced_dim, self.extractor_dim)
            self.tail_extractor = nn.Linear(2 * self.reduced_dim, self.extractor_dim)
        else:
            self.cls_info_head_extractor = nn.Linear(3 * self.reduced_dim, self.extractor_dim)
            self.cls_info_tail_extractor = nn.Linear(3 * self.reduced_dim, self.extractor_dim)

        if self.use_bilinear:
           
            self.bilinear = nn.Linear(self.extractor_dim * block_size, config.num_labels)
        else:
           
           
            self.classifier = nn.Parameter(
                torch.randn(self.Relation_Specific_num, self.reduced_dim * 2, self.reduced_dim * 2))
           
            self.classifier_bais = nn.Parameter(torch.randn(self.Relation_Specific_num))

            nn.init.uniform_(self.classifier, a=-math.sqrt(1 / (2 * self.reduced_dim)),
                             b=math.sqrt(1 / (2 * self.reduced_dim)))
            nn.init.uniform_(self.classifier_bais, a=-math.sqrt(1 / (2 * self.reduced_dim)),
                             b=math.sqrt(1 / (2 * self.reduced_dim)))
       
        self.relation_vector = nn.Parameter(torch.randn(self.Relation_Specific_num, self.relation_dim))
        nn.init.xavier_normal_(self.relation_vector)

       
        self.SAISLoss = SAISLoss()
       
        self.ET_predictor_module = nn.Linear(reduced_dim, Relation_Specific_num+1)

        with open('env_sent.pkl', 'wb') as f:
            pickle.dump([], f)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
       
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens,
                                                        self.three_atten)

        return sequence_output, attention

    def get_hrt(self, sequence_output, cls_mask, attention, entity_pos, hts, offset, tag):
        hss, tss, rss, Crs, ht_atts, bs_ht_num = Rel_Mutil_men2ent(sequence_output, cls_mask, attention, entity_pos,
                                                                   hts, offset, self.relation_vector, self.reduced_dim,
                                                                   self.Relation_Specific_num, self.dropout, tag)
        rels_per_batch = [len(b) for b in hss] 
        hss = torch.cat(hss, dim=0) 
        tss = torch.cat(tss, dim=0) 
        rss = torch.cat(rss, dim=0) 
        ht_atts = torch.cat(ht_atts, dim=0) 

        return hss, rss, tss, Crs, bs_ht_num, ht_atts, rels_per_batch

    def forward_rel(self, hs, ts, rs):
        if not self.use_cls_info:
            hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
            ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))
        else:
           
            hs = torch.tanh(self.cls_info_head_extractor(torch.cat([hs, rs], dim=-1)))
            ts = torch.tanh(self.cls_info_tail_extractor(torch.cat([ts, rs], dim=-1)))

       
        b1 = hs.view(-1, hs.shape[1], self.extractor_dim // self.block_size, self.block_size)
        b2 = ts.view(-1, ts.shape[1], self.extractor_dim // self.block_size, self.block_size)
        bl = (b1.unsqueeze(4) * b2.unsqueeze(3)).view(-1, hs.shape[1], self.extractor_dim * self.block_size)
        logits = self.bilinear(bl) 

       
        SCL_bl = bl.mean(1)

        logits = logits.mean(1) 
       

        return logits, SCL_bl

    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)): 
           
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i + 1])]
           
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]] 

           
            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
           
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
           
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1) 
            rel_sent_attn.append(sum_attn)

       
        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn

    def get_hrt_logsumexp(self, sequence_output, attention, entity_pos, hts, offset):
        n, h, _, c = attention.size()
        ht_atts = []

        for i in range(len(entity_pos)): 
            entity_atts = []
            for eid, e in enumerate(entity_pos[i]): 
                if len(e) > 1:
                    e_att = []
                    for mid, (start, end) in enumerate(e): 
                        if start + offset < c:
                            e_att.append(attention[i, :, start + offset])

                    if len(e_att) > 0:
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_att = attention[i, :, start + offset]
                    else:
                        e_att = torch.zeros(h, c).to(attention)

                entity_atts.append(e_att)

            entity_atts = torch.stack(entity_atts, dim=0)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1) 
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)

        rels_per_batch = [len(b) for b in ht_atts]
        ht_atts = torch.cat(ht_atts, dim=0)

        return ht_atts, rels_per_batch


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None, 
                entity_pos=None,
                hts=None, 
                sent_pos=None,
                sent_labels=None, 
                teacher_attns=None, 
                tag="train",
                epair_types=None,
                ):

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)

        if tag in enhance_evi:
            doc_attn_logsumexp, batch_rel_logsumexp = self.get_hrt_logsumexp(sequence_output, attention, entity_pos, hts, offset)
       
        hs, rs, ts, Crs, bs_ht_num, doc_attn, batch_rel = self.get_hrt(sequence_output, attention_mask, attention, entity_pos, hts, offset, tag)
       
        doc_attn = doc_attn.mean(1)

        if TASK_ET:
            batch_ET_reps = self.SAISLoss.get_ET_reps([hs, ts])
            batch_ET_reps = self.ET_predictor_module(batch_ET_reps)
            batch_ET_reps = torch.diagonal(batch_ET_reps, dim1=-2, dim2=-1) 

           
            top_values, top_indices = torch.topk(batch_ET_reps, k=Top_k, dim=1)
            mask = torch.zeros_like(batch_ET_reps).to(batch_ET_reps)
            mask.scatter_(1, top_indices, 1) 
            h_mask, t_mask = torch.chunk(mask.bool().unsqueeze(2).repeat(1, 1, hs.shape[2]), chunks=2, dim=0)

           
            hs_new = torch.masked_select(hs, h_mask).reshape(hs.shape[0], Top_k, hs.shape[2])
            ts_new = torch.masked_select(ts, t_mask).reshape(ts.shape[0], Top_k, ts.shape[2])
            if TSER_Logsumexp:
                hs = torch.cat((torch.mean(hs, dim=1, keepdim=True), hs_new), dim=1) 
                ts = torch.cat((torch.mean(ts, dim=1, keepdim=True), ts_new), dim=1)
            else: 
               
               
                hs = torch.mean(hs, dim=1, keepdim=True)
                ts = torch.mean(ts, dim=1, keepdim=True)

            ht_mask = (h_mask | t_mask)[:, :, 0].sum(dim=1, keepdim=True).unsqueeze(2).repeat(1, 1, rs.shape[2]) 
            rs_new = rs.sum(dim=1, keepdim=True) / ht_mask
            if TSER_Logsumexp:
                rs = torch.cat((torch.mean(rs, dim=1, keepdim=True), rs_new), dim=1) 
            else:
                rs = torch.mean(rs, dim=1, keepdim=True)

       
        logits, SCL_bl = self.forward_rel(hs, ts, rs)
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)

        if sent_labels != None:
            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
            if tag in enhance_evi:
                s_attn_logsumexp = self.forward_evi(doc_attn_logsumexp, sent_pos, batch_rel_logsumexp, offset)
                s_attn = s_attn_lambda * s_attn + s_attn_logsumexp_lambda * s_attn_logsumexp

            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))
       
        if tag in ["test", "dev", 'train']: 
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels) 
            output["scores"] = scores_topk[0] 
            output["topks"] = scores_topk[1] 

        if tag == "infer": 
            output["attns"] = doc_attn.split(batch_rel)

        else:
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}
           

            if sent_labels != None:
                idx_used = torch.nonzero(labels[:, 1:].sum(dim=-1)).view(-1) 
               
                s_attn = s_attn[idx_used]
                s_attn[s_attn == 0] = 1e-30

               
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30

               
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels) 
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)

            elif teacher_attns != None: 
                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)

            if PEMSCLloss:
                scl_loss = self.SCL_loss(F.normalize(SCL_bl, dim=-1), labels)
                output["loss"]["pemscl_loss"] = scl_loss * self.lambda_3

            if Relation_Seploss:
                self.relation_seploss = self.Rel_Spe_loss(Crs, hs * ts, bs_ht_num, self.relation_vector)

                output["loss"]["relation_seploss"] = self.relation_seploss * self.lambda_3

            if TASK_ET:
                etoss = self.SAISLoss.cal_ET_loss(batch_ET_reps, epair_types)
                output["loss"]["etoss"] = etoss * loss_weight_ET

        return output
