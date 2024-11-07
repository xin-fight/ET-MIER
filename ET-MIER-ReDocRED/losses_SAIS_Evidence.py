import torch
import torch.nn as nn
from constant_SAIS_Evidence import *
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)  # theshold is norelation
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]  # smallest logits among the num_labels
            # predictions are those logits > thresh and logits >= smallest
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        # if no such relation label exist: set its label to 'Nolabel'
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    def get_score(self, logits, num_labels=-1):
        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:, 1] - logits[:, 0], 0


class SAISLoss:

    def __init__(self):
        pass

    def cal_ET_loss(self, batch_ET_reps, batch_epair_types):
        batch_epair_types = batch_epair_types.T.flatten()
        batch_ET_loss = F.cross_entropy(batch_ET_reps, batch_epair_types-1)

        return batch_ET_loss

    def get_ET_reps(self, batch_epair_reps):
        batch_ET_reps = torch.tanh(torch.cat(batch_epair_reps, dim=0)).to(batch_epair_reps[0])

        return batch_ET_reps


class MLLTRSCLloss(nn.Module):
    def __init__(self, tau=2.0, tau_base=1.0):
        super().__init__()
        self.tau = tau
        self.tau_base = tau_base

    def forward(self, features, labels, weights=None):
        labels = labels.long()

        label_mask = (labels[:, 0] != 1.)

        mask_s = torch.any((labels.unsqueeze(1) & labels).bool(), dim=-1).float().fill_diagonal_(0)

        sims = torch.div(features.mm(features.T), self.tau)

        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()

        logits_mask = torch.ones_like(mask_s).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        denom = mask_s.sum(1)
        denom[denom == 0] = 1
        log_prob1 = (mask_s * log_prob).sum(1) / denom

        log_prob2 = - torch.log(exp_logits.sum(-1, keepdim=True)) * (mask_s.sum(-1, keepdim=True) == 0)

        mean_log_prob_pos = (log_prob1 + log_prob2) * label_mask

        loss = - (self.tau / self.tau_base) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class Relation_Specificloss(nn.Module):
    def __init__(self, tau=2.0, tau_base=1.0):
        super().__init__()
        self.tau = tau
        self.tau_base = tau_base

    def logsumexploss(self, emb, mask):

        sims = torch.div(emb.mm(emb.T), self.tau)

        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()

        logits_mask = torch.ones_like(mask).fill_diagonal_(0).to(emb)

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        denom = mask.sum(1)
        denom[denom == 0] = 1

        log_prob1 = (mask * log_prob).sum(1) / denom

        loss = - (self.tau / self.tau_base) * log_prob1
        loss = loss.mean()

        return loss

    def MY_logsumexploss(self, emb, mask):
        sims = torch.div(emb.mm(emb.T), self.tau)

        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()

        logits_mask = torch.ones_like(mask).fill_diagonal_(0).to(emb)
        exp_logits_htd = torch.exp(logits) * logits_mask

        exp_logits_htp = torch.exp(logits) * mask

        denom = mask.sum(1, keepdim=True)
        denom[denom == 0] = 1

        log_prob = torch.log(exp_logits_htp.sum(1, keepdim=True)) - torch.log(
            exp_logits_htd.sum(1, keepdim=True)) - torch.log(denom)

        loss = - (self.tau / self.tau_base) * log_prob
        loss = loss.mean()

        return loss

    def forward(self, Crs, emb, bs_ht_num, relation_vector):
        loss = 0

        if Crht_loss:
            Relation_Specific_num, _ = Crs[0].shape
            bs = len(Crs)

            Crs = torch.cat(Crs, dim=0).to(emb)

            unit_matrix = torch.eye(Relation_Specific_num)
            left_right_concat = torch.cat([unit_matrix] * bs, dim=1)
            mask_s = torch.cat([left_right_concat] * bs, dim=0).float().fill_diagonal_(0).to(emb)

            loss_Crs = self.MY_logsumexploss(Crs, mask_s)
            del mask_s

            loss += loss_Crs

        if Emb_loss:
            num_e, Relation_Specific_num, reduced_dim = emb.shape
            emb = emb.view(num_e * Relation_Specific_num, reduced_dim)

            unit_matrix = torch.eye(Relation_Specific_num)
            left_right_concat = torch.cat([unit_matrix] * num_e, dim=1)
            mask_e = torch.cat([left_right_concat] * num_e, dim=0).float().fill_diagonal_(0).to(emb)

            loss_emb = self.MY_logsumexploss(emb, mask_e)
            del mask_e

            loss += loss_emb

        if Relation_loss:
            relation_vector_num, _ = relation_vector.shape
            sims = torch.div(relation_vector.mm(relation_vector.T), self.tau)

            logits_max, _ = torch.max(sims, dim=1, keepdim=True)
            logits = sims - logits_max.detach()

            logits_mask = torch.ones(relation_vector_num, relation_vector_num).fill_diagonal_(0).to(emb)
            exp_logits_htd = torch.exp(logits) * logits_mask

            log_prob = exp_logits_htd.sum(1, keepdim=True)

            # loss += (self.tau / self.tau_base) * log_prob.mean()
            from constant_SAIS_Evidence import Relation_loss_lambda
            loss += Relation_loss_lambda * log_prob.mean()

        return loss
