import torch
import torch.nn as nn
from models.dgcnn import DGCNN

class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128): # change hidden_size to 256 later
        super().__init__() 

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        
        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
            nn.ReLU()
        )
        # self.match = nn.Conv1d(hidden_size, 1, 1)

        self.graph = DGCNN(
            input_dim=128, #change later to 256
            output_dim=128, #change later to 256
            k=20
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
        
        # DGCNN
        graph_output = self.graph(features) # batch_size, hidden_size, num_proposals

        # fuse
        features = torch.cat([graph_output, lang_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
        features = features.permute(0, 2, 1).contiguous() # batch_size, 128 + lang_size, num_proposals

        # fuse features
        features = self.fuse(features) # batch_size, hidden_size, num_proposals
        
        # mask out invalid proposals
        objectness_masks = objectness_masks.permute(0, 2, 1).contiguous() # batch_size, 1, num_proposals
        features = features * objectness_masks # batch_size, hidden_size, num_proposals

        # match
        confidences = self.match(features).squeeze(1) # batch_size, num_proposals
                
        data_dict["cluster_ref"] = confidences

        return data_dict
