import torch
from torch import nn
from encoder import encoder_layer
from position_encoding import transformer_embeding

class TaxtClassification(nn.Module):
    def __init__(
            self, vocab_size, d_model,
            num_heads, num_layers, num_classes, 
            max_len, dropout, device
            ):
        super(TaxtClassification, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device

        self.embedding = transformer_embeding(vocab_size, d_model, max_len, device, dropout)
        self.encoder_layers = nn.ModuleList([
            encoder_layer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_embedded = self.embedding(x)
        encoder_output = x_embedded

        cls_output = encoder_output[:, 0, :]
        output = self.classifier(cls_output)
        return self.softmax(output)

        