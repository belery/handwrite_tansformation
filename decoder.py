import torch
from torch import nn
from head_attention import mutihead_attention
from position_wise_feed_forwar import position_wise_feed_forward


class decoder_layer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super (decoder_layer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attention = mutihead_attention(d_model, num_heads)
        self.ffn_hidden = position_wise_feed_forward(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)


    def forward(self, x, encoder_output, encoder_mask):
        temp_x = x
        x = self.attention.forward(x, x, x)
        x = self.dropout(x)
        x = self.norm1(temp_x + x)
        temp_x = x
        

        