from torch import nn
import torch


if __name__=='__main__':
    conv1 = nn.Conv2d(1, 512,(8,64))
    input = torch.randn(512,128,1200)  # [batch_size, max_len, embedding_dim]
    input = torch.unsqueeze(input, 1)
    input = input.permute(0, 1, 3, 2)
    print(input.shape)
    out = conv1(input)                # [batch_size, out_channels, n+2p-f/s+1]
    print(out.shape)
