import torch
from torch import nn
from position_encoding import TokenEmbedding, positional_encoding, transformer_embeding
from encoder import encoder_layer
from decoder import decoder_layer
from Chinese_tokenizer import ChineseTransformer

class transformer(nn.Module):
    def __init__(self, d_model=512, n_head=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_len=100, device='cpu', vocab_size=10000):
        super(transformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head  
        self.ChineseTransformer = ChineseTransformer(vocab_size = 10000, d_model = 512, max_len = 3000, dropout = 0.1)  
        with open('cnews/cnews.test.txt', 'r', encoding='utf-8') as f:
            self.train_text = f.read()

        with open('cnews/cnews.val.txt', 'r', encoding='utf-8') as f:
            self.test_text = f.read()
        self.tokenizer_test = self.ChineseTransformer.train_tokenizer(self.test_text)
        self.tokenizer_train = self.ChineseTransformer.train_tokenizer(self.train_text)
        self.encoder_layers = nn.ModuleList([
            encoder_layer(d_model=d_model, num_heads=n_head, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            decoder_layer(d_model=d_model, num_heads=n_head, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

    def forward(self):
        encoder_output = self.ChineseTransformer.train_tokenizer(self.train_text)
        i = 1
        for encoder_layer in self.encoder_layers:
            print("-----------encoder_layer------------ %d" % i)
            i += 1
            encoder_output = encoder_layer(encoder_output)
        
        # 解码器部分 - 重复应用n次
        i = 1
        decoder_output = self.ChineseTransformer.train_tokenizer(self.test_text)
        for decoder_layer in self.decoder_layers:
            print("-----------decoder_layer------------ %d" % i)
            decoder_output = decoder_layer(decoder_output, encoder_output)
            i += 1
        print(ChineseTransformer.preprocess(encoder_output), ChineseTransformer.preprocess(decoder_output))

if __name__ == '__main__':
    model = transformer()
    model.forward()
    