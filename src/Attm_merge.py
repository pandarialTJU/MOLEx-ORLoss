'''
The AttM module implementation for merging the multiple layer outputs from SSL models
to a single feature representation.
Reference: Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection
Source: https://github.com/pandarialTJU/AttM_INTERSPEECH24.git
Author: Zihan Pan
'''

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import ipdb


def str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False


class AttM(nn.Module):

    # input x is the multiple outputs from each transform layer of SSL models
    # 12 or 24 layers from WavLM, HuBert, Wav2Vec2

    def __init__(self, d_args) -> None:
        super(AttM, self).__init__()

        self.d_args = d_args
        self.n_feat = self.d_args["SSL_dim"]
        self.n_layer = self.d_args["SSL_layer_num"]

        self.W = nn.Parameter(torch.randn(self.n_feat, 1))
        self.W1 = nn.Parameter(torch.randn(self.n_layer, int(self.n_layer/2)))
        self.W2 = nn.Parameter(torch.randn(int(self.n_layer/2), self.n_layer))
        self.hidden = int(self.n_layer*self.n_feat/4)
        self.linear_proj = nn.Sequential(
            nn.Linear(self.n_layer*self.n_feat, self.hidden),
            nn.SiLU(nn.Linear(self.hidden, self.hidden)),
            nn.Linear(self.hidden, self.n_feat),
        )
        self.SWISH = nn.SiLU()

    
    def forward(self, x):

        # X1 has dimension of Bs x Layers x Time x Hidden feature

        x_input = x
   
        x = torch.mean(x, dim=2, keepdim=True) # X2 = AVG(X1) AVG across time dim

        x = self.SWISH(torch.matmul(x, self.W)) # X3

        x = self.SWISH(torch.matmul(x.view(-1, self.n_layer), self.W1))
        x = torch.sigmoid((torch.matmul(x, self.W2))) # X4
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.mul(x, x_input) # X5

        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2), -1) # concatenate

        x = self.linear_proj(x)

        return x



class LinM(nn.Module):

    # positve weight sum to SSL model

    def __init__(self, d_args) -> None:
        super(LinM, self).__init__()

        self.d_args = d_args
        self.layer_num = d_args["SSL_layer_num"]
        self.W = nn.Parameter(torch.zeros(self.layer_num, 1))
        

    def forward(self, x):
        # x is the 24-layer maps: [bs, 24, T, H]

        x = torch.matmul(x.permute(0,2,3,1), F.softmax(self.W, dim=0))

        return x.squeeze()


