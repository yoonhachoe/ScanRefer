import torch
import torch.nn as nn
from models.dgcnn import DGCNN


class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=256, use_cross_attn=False, use_dgcnn=False):
        super().__init__()
        # hidden_size=128
        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.use_dgcnn = use_dgcnn
        self.use_cross_attn = use_cross_attn

        #self.cross1 = nn.Linear(self.lang_size, hidden_size)
        #self.cross2 = nn.Linear(self.hidden_size+self.lang_size, hidden_size)

        self.skip = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, self.hidden_size, 1),
        )

        self.graph = DGCNN(
            input_dim=self.lang_size + 128,
            output_dim=self.hidden_size,
            k=6
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, self.hidden_size, 1),
            nn.ReLU()
        )

        self.cross = nn.Linear(self.lang_size, self.lang_size)

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

        # DGCNN
        if self.use_dgcnn:
            # fuse
            features = torch.cat([features, lang_feat], dim=-1)  # batch_size, num_proposals, 128 + lang_size
            features = features.permute(0, 2, 1).contiguous()  # batch_size, 128 + lang_size, num_proposals
            # mask out invalid proposals
            objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
            features = features * objectness_masks  # batch_size, 128 + lang_size, num_proposals
            skipfeatures = self.skip(features)  # batch_size, hidden_size, num_proposals
            features = self.graph(features) + skipfeatures  # batch_size, hidden_size, num_proposals
        else: #no graph
            # fuse
            features = torch.cat([features, lang_feat], dim=-1)  # batch_size, num_proposals, 128 + lang_size
            features = features.permute(0, 2, 1).contiguous()  # batch_size, 128 + lang_size, num_proposals
            # fuse features
            features = self.fuse(features)  # batch_size, 128, num_proposals
            # mask out invalid proposals
            objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
            features = features * objectness_masks  # batch_size, 128, num_proposals

        if self.use_cross_attn:
            features = features.permute(0, 2, 1).contiguous()  # batch_size, num_proposals, hidden_size
            _, P, _ = features.size()
            lang_token = data_dict["attn_value"] # batch_size, timestep, lang_size
            _, T, _ = lang_token.size()
            features = torch.cat([features, lang_token], dim=1)  # batch_size, num_proposals + timestep, hidden_size
            score = self.cross(features) # batch_size, num_proposals + timestep, lang_size
            score = torch.bmm(score, features.permute(0, 2, 1).contiguous()) # batch_size, num_proposals + timestep, num_proposals + timestep
            weight = nn.functional.softmax(score, dim=2)
            value = torch.bmm(weight, features) # batch_size, num_proposals + timestep, hidden_size
            final_features = value[:,[0,P-1],:] # b, p, hidden_size
            final_lang = value[:,[P,P+T-1],:] # b, t, lang_size
            final_lang, _ = torch.max(final_lang, 1) # b, lang_size
            print(final_lang.size())
            #match
            confidences = torch.bmm(final_features, final_lang.unsqueeze(2)) # b, p
            #confidences = self.match(value).squeeze(1) # batch_size, num_proposals
        else:
             # match
            confidences = self.match(features).squeeze(1) # batch_size, num_proposals

        data_dict["cluster_ref"] = confidences

        return data_dict
