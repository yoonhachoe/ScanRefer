import torch
import torch.nn as nn
from models.dgcnn import DGCNN


class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, use_cross_attn=False, use_dgcnn=False, fuse_before=False):
        super().__init__() 

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.use_dgcnn = use_dgcnn
        self.use_cross_attn = use_cross_attn
        self.fuse_before = fuse_before

        self.cross1 = nn.Linear(self.lang_size, hidden_size)
        self.cross2 = nn.Linear(self.hidden_size+self.lang_size, hidden_size)

        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
            nn.ReLU()
        )

        self.fuse2 = nn.Sequential(
            nn.Conv1d(self.hidden_size*2, hidden_size, 1),
            nn.ReLU()
        )

        self.graph = DGCNN(
            input_dim=self.lang_size + 3 + 128,
            output_dim=128,
            k=6
        )

        self.skip = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
        )

        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # unpack outputs from detection branch
        features = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"] # batch_size, lang_size
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size, num_proposals, lang_size

        center = data_dict['center']  # batch_size, num_proposal, 3
        center = center.permute(0, 2, 1).contiguous()  # batch_size, 3, num_proposal

        # DGCNN
        if self.use_dgcnn:
            # fuse
            #features = torch.cat([features, lang_feat], dim=-1)  # batch_size, num_proposals, 128 + lang_size
            features = features.permute(0, 2, 1).contiguous()  # batch_size, 128, num_proposals
            # mask out invalid proposals
            objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
            features = features * objectness_masks  # batch_size, 128, num_proposals
            skipfeatures = features  # batch_size, 128, num_proposals
            features = torch.cat([features, center], dim=1)  # batch_size, 128 + 3, num_proposals
            features = self.graph(features) + skipfeatures  # batch_size, 128, num_proposals

        else: #no graph
            if not self.use_cross_attn: #no cross-attention (same as scanrefer)
                # fuse
                features = torch.cat([features, lang_feat], dim=-1)  # batch_size, num_proposals, 128 + lang_size
                features = features.permute(0, 2, 1).contiguous()  # batch_size, 128 + lang_size, num_proposals
                # fuse features
                features = self.fuse(features)  # batch_size, hidden_size, num_proposals

        if self.use_cross_attn:
            objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
            lang_cross = data_dict["attn_value"] # batch_size, timestep, lang_size
            lang_cross = self.cross1(lang_cross) # batch_size, timestep, hidden_size
            features_cross = self.cross2(features) # batch_size, num_proposals, hidden_size
            score = torch.bmm(features_cross, lang_cross.permute(0, 2, 1).contiguous()) # batch_size, num_proposals, timestep
            weight = nn.functional.softmax(score, dim=2)
            value = torch.bmm(weight, lang_cross) # batch_size, num_proposals, hidden_size
            value = value + features # batch_size, num_proposals, hidden_size
            value = value.permute(0, 2, 1).contiguous() # batch_size, hidden_size, num_proposals
            #value = torch.cat([value, features], dim=-1) # batch_size, num_proposals, 2*hidden_size
            #value = self.fuse2(value.permute(0, 2, 1).contiguous()) # batch_size, hidden_size, num_proposals
            value = value * objectness_masks
            #match
            confidences = self.match(value).squeeze(1) # batch_size, num_proposals
        else:
            features = features * objectness_masks
             # match
            confidences = self.match(features).squeeze(1) # batch_size, num_proposals

        data_dict["cluster_ref"] = confidences

        return data_dict
