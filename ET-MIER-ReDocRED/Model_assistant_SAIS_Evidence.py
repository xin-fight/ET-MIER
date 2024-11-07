import torch.nn as nn
from opt_einsum import contract
from constant_SAIS_Evidence import *


def Use_Cls_Info(cls_mask, sequence_output, Relation_Specific_num):
    keep_mask = cls_mask.unsqueeze(-1)
    cls_rep = sequence_output * keep_mask
    cls_rep = torch.mean(cls_rep, dim=1, keepdim=True)
    cls_info = cls_rep.repeat(1, Relation_Specific_num + 1, 1)
    return cls_info


def Rel_Mutil_men2ent(sequence_output, cls_mask, attention, entity_pos, hts, offset, relation_vector, reduced_dim,
                      Relation_Specific_num, dropout, tag):
    n, h, _, c = attention.size()
    hss, tss, rss, Crs = [], [], [], []
    ht_atts, bs_ht_num = [], [0]

    if use_cls_info:
        cls_info = Use_Cls_Info(cls_mask, sequence_output, Relation_Specific_num)

    for i in range(len(entity_pos)):
        entity_embs, entity_atts = [], []
        # obtain entity embedding from mention embeddings.
        for eid, e in enumerate(entity_pos[i]):  # for each entity
            if len(e) > 1:
                e_emb, e_att = [], []
                for start, end in e:
                    if start + offset < c:
                        e_emb.append(sequence_output[i, start + offset])
                        e_att.append(attention[i, :, start + offset])

                if len(e_emb) > 0:
                    e_emb_logsumexp = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0).unsqueeze(0)
                    e_att_mean = torch.stack(e_att, dim=0).mean(0).unsqueeze(0)

                    men_rep = torch.stack(e_emb, dim=0)

                    men_attention = torch.matmul(nn.Tanh()(men_rep), relation_vector.transpose(0, 1).contiguous())
                    men_attention = nn.Softmax(dim=0)(men_attention).permute(1, 0).contiguous()

                    e_emb = torch.matmul(men_attention, men_rep)

                    e_emb = torch.cat((e_emb_logsumexp, e_emb), dim=0)  # [Relation_Specific_num+1, reduced_dim]

                    e_att = torch.stack(e_att, dim=0)

                    e_att = torch.matmul(men_attention, e_att.reshape(men_attention.shape[1], -1)).reshape(
                        Relation_Specific_num, h, -1)

                    e_att = torch.cat((e_att_mean, e_att), dim=0)

                else:
                    e_emb = torch.zeros(Relation_Specific_num + 1, reduced_dim).to(sequence_output)
                    e_att = torch.zeros(Relation_Specific_num + 1, h, c).to(attention)
            else:
                start, end = e[0]
                if start + offset < c:
                    e_emb = sequence_output[i, start + offset]
                    e_att = attention[i, :, start + offset]

                    e_emb_o = e_emb.unsqueeze(0)
                    e_att_o = e_att.unsqueeze(0)

                    if one_mention_copy_or_addrel:
                        e_emb = e_emb.expand(Relation_Specific_num, -1)
                    else:
                        men_rep = e_emb_o + relation_vector
                        men_attention = torch.matmul(nn.Tanh()(men_rep), relation_vector.transpose(0, 1).contiguous())
                        men_attention = nn.Softmax(dim=0)(men_attention).permute(1, 0).contiguous()

                        e_emb = torch.matmul(men_attention, men_rep)
                    e_emb = torch.cat((e_emb_o, e_emb), dim=0)

                    if one_mention_copy_or_addrel:
                        e_att = e_att.expand(Relation_Specific_num, h, -1)
                    else:
                        e_att = e_att.expand(Relation_Specific_num, h, -1)

                        if tag in ["train", "test"]:
                            p = torch.bernoulli(torch.rand_like(e_att))
                            e_att = e_att * p

                        e_att = torch.matmul(men_attention, e_att.reshape(men_attention.shape[1], -1)).reshape(
                            Relation_Specific_num, h, -1)
                    e_att = torch.cat((e_att_o, e_att), dim=0)

                else:
                    e_emb = torch.zeros(Relation_Specific_num + 1, reduced_dim).to(sequence_output)
                    e_att = torch.zeros(Relation_Specific_num + 1, h, c).to(attention)

            if use_cls_info:
                e_emb = torch.cat([e_emb, cls_info[i]], dim=-1)
                e_emb = dropout(e_emb)
                e_att = dropout(e_att)

            entity_embs.append(e_emb)
            entity_atts.append(e_att)

        entity_embs = torch.stack(entity_embs, dim=0)
        entity_atts = torch.stack(entity_atts, dim=0)

        ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
        hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
        ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

        h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
        t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

        ht_att = (h_att * t_att).mean(2)
        ht_att = ht_att / (ht_att.sum(2, keepdim=True) + 1e-30)

        rs = contract("ld,nal->nad", sequence_output[i], ht_att)

        rs_Csor = rs[:, 1:, :]

        Cso_attention = (nn.Tanh()(rs_Csor) * relation_vector.unsqueeze(0)).sum(dim=2)

        Cso_attention = nn.Softmax(dim=0)(Cso_attention).permute(1, 0).contiguous()
        Cr_rep_spec = torch.matmul(Cso_attention, rs_Csor.reshape(rs_Csor.shape[0], -1)).reshape(Relation_Specific_num,
                                                                                                 Relation_Specific_num,
                                                                                                 -1)

        Cr_rep_spec = Cr_rep_spec.permute(2, 0, 1)
        Cr_rep_spec = torch.diagonal(Cr_rep_spec, dim1=-2, dim2=-1)
        Cr_rep_spec = Cr_rep_spec.permute(1, 0)

        hss.append(hs)
        tss.append(ts)
        rss.append(rs)
        Crs.append(Cr_rep_spec)
        ht_atts.append(ht_att)

        bs_ht_num.append(bs_ht_num[-1] + len(hs))

    return hss, tss, rss, Crs, ht_atts, bs_ht_num


def Mutil_men2ent_O(entity_pos, sequence_output, attention, offset, hts, config):
    n, h, _, c = attention.size()
    hss, tss, rss = [], [], []
    ht_atts = []

    for i in range(len(entity_pos)):  # for each batch
        entity_embs, entity_atts = [], []

        # obtain entity embedding from mention embeddings.
        for eid, e in enumerate(entity_pos[i]):  # for each entity
            if len(e) > 1:
                e_emb, e_att = [], []
                for mid, (start, end) in enumerate(e):  # for every mention
                    if start + offset < c:
                        # In case the entity mention is truncated due to limited max seq length.
                        e_emb.append(sequence_output[i, start + offset])
                        e_att.append(attention[i, :, start + offset])

                if len(e_emb) > 0:
                    e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                    e_att = torch.stack(e_att, dim=0).mean(0)
                else:
                    e_emb = torch.zeros(config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(h, c).to(attention)
            else:
                start, end = e[0]
                if start + offset < c:
                    e_emb = sequence_output[i, start + offset]
                    e_att = attention[i, :, start + offset]
                else:
                    e_emb = torch.zeros(config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(h, c).to(attention)

            entity_embs.append(e_emb)
            entity_atts.append(e_att)

        entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
        entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

        ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

        # obtain subject/object (head/tail) embeddings from entity embeddings.
        hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
        ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

        h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
        t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

        ht_att = (h_att * t_att).mean(1)  # average over all heads
        ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
        ht_atts.append(ht_att)

        # obtain local context embeddings.
        rs = contract("ld,rl->rd", sequence_output[i], ht_att)

        hss.append(hs)
        tss.append(ts)
        rss.append(rs)

    return hss, tss, rss, ht_atts
