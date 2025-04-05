# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn

from recbole_gnn.model.general_recommender.lightgcn import LightGCN


class WeightedGCL(LightGCN):
    def __init__(self, config, dataset):
        super(WeightedGCL, self).__init__(config, dataset)

        self.latent_dim = config['embedding_size']
        self.cl_rate = config['lambda']
        self.eps = config['eps']
        self.temperature = config['temperature']
        self.k = config['k']
        # self.method = config['method']

        # # 1 -> 64
        # self.excitation = nn.Sequential(
        #     nn.Linear(1, self.latent_dim),
        #     nn.Sigmoid()
        # )

        # # 1 -> 8 -> 64
        # self.excitation = nn.Sequential(
        #     nn.Linear(1, (int)(self.latent_dim / 8)),
        #     nn.Linear((int)(self.latent_dim / 8), self.latent_dim),
        #     nn.Sigmoid()
        # )

        # 1 -> 4 -> 16 -> 64
        self.excitation = nn.Sequential(
            nn.Linear(1, (int)(self.latent_dim / 16)),
            nn.Linear((int)(self.latent_dim / 16), (int)(self.latent_dim / 4)),
            nn.Linear((int)(self.latent_dim / 4), self.latent_dim),
            nn.Sigmoid()
        )

        # # 1 -> 2 -> 4 -> 16 -> 64
        # self.excitation = nn.Sequential(
        #     nn.Linear(1, (int)(self.latent_dim / 32)),
        #     nn.Linear((int)(self.latent_dim / 32), (int)(self.latent_dim / 16)),
        #     nn.Linear((int)(self.latent_dim / 16), (int)(self.latent_dim / 4)),
        #     nn.Linear((int)(self.latent_dim / 4), self.latent_dim),
        #     nn.Sigmoid()
        # )

    def forward(self, perturbed=False):
        all_embs = self.get_ego_embeddings()
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            if perturbed and layer_idx >= self.n_layers - 1:
            # if perturbed:
                random_noise = torch.rand_like(all_embs, device=all_embs.device)
                all_embs = all_embs + torch.sign(all_embs) * F.normalize(random_noise, dim=-1) * self.eps
            embeddings_list.append(all_embs)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def weight_features(self, view1, view2):
        # squeeze
        mid_view1, mid_view2 = torch.mean(view1, dim=1, keepdim=True), torch.mean(view2, dim=1, keepdim=True)
        # excitation
        result1, result2 = self.excitation(mid_view1), self.excitation(mid_view2)
        return result1 * view1, result2 * view2

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()

    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)

        user = torch.unique(interaction[self.USER_ID])
        pos_item = torch.unique(interaction[self.ITEM_ID])

        perturbed_user_embs_1, perturbed_item_embs_1 = self.forward(perturbed=True)
        perturbed_user_embs_2, perturbed_item_embs_2 = self.forward(perturbed=True)

        user_view1, user_view2 = self.weight_features(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_view1, item_view2 = self.weight_features(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])


        user_cl_loss = self.calculate_cl_loss(user_view1, user_view2)
        item_cl_loss = self.calculate_cl_loss(item_view1, item_view2)

        return loss, self.cl_rate * (user_cl_loss + item_cl_loss)
