import torch
import torch.nn as nn
import utils
import dataset

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder_layer, decoder_layer):
        super(Encoder_Decoder,self).__init__()
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
    
    def forward(self,x):
        self.encoder_layer_output = self.encoder_layer(x)
        self.decoder_layer_output = self.decoder_layer(self.encoder_layer_output)
        return self.decoder_layer_output

class Encoder_layer(nn.Module):
    def __init__(self, first_encoder, sequential_encoder=None, depth=2):
        super(Encoder_layer,self).__init__()
        self.first_encoder = first_encoder
        self.depth = depth
        if self.depth >= 2:
            self.sequential_encoders = utils.clones(sequential_encoder,depth-1)
    def forward(self,x):
        x = self.first_encoder(x)
        if self.depth >= 2:
            for encoder in self.sequential_encoders:
                x = encoder(x)
        return x
    
class First_Encoder(nn.Module):
    def __init__(self, W_s, C, f_c, activation):
        super(First_Encoder, self).__init__()
        self.linear = nn.Linear(W_s*C, f_c)
        self.activation = activation
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class Sequential_Encoder(nn.Module):
    def __init__(self, f_c, activation):
        super(Sequential_Encoder, self).__init__()
        self.linear = nn.Linear(f_c, f_c)
        self.activation = activation
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
    
class Decoder_layer(nn.Module):
    def __init__(self, first_decoder, sequential_decoder=None, depth=2):
        super(Decoder_layer,self).__init__()
        self.first_decoder = first_decoder
        self.depth = depth
        if self.depth >= 2:
            self.sequential_decoders = utils.clones(sequential_decoder,depth-1)
    def forward(self,x):
        x = self.first_decoder(x)
        if self.depth >= 2:
            for decoder in self.sequential_decoders:
                x = decoder(x)
        return x

class First_Decoder(nn.Module):
    def __init__(self, W_s, C, f_c, activation):
        super(First_Decoder, self).__init__()
        self.linear = nn.Linear(f_c,W_s*C)
        self.activation = activation
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
    
class Sequential_Decoder(nn.Module):
    def __init__(self, W_s, C, activation):
        super(Sequential_Decoder, self).__init__()
        self.linear = nn.Linear(W_s*C, W_s*C)
        self.activation = activation
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

if __name__ =='__main__':
    # hyperparameters
    W_s = 10
    C= 4
    f_c = 5
    # get some data
    data_dir = './datasets/SVC2004/task1/training/'
    train_data = dataset.SVC2004(data_dir, 793, 10)
    print(train_data[0].shape)
    # define some model

    # 1.1 first_encoder.
    encoder = First_Encoder(W_s,C,f_c, nn.Sigmoid())
    #encoder.double()
    print(encoder(train_data[0]))
    # 1.2 sequential_encoder
    sequential_encoder = Sequential_Encoder(f_c, nn.Sigmoid())
    # 1.3 encoder_layer
    encoder_layer = Encoder_layer(encoder, sequential_encoder, 2)
    # 1.4 first_decoder.
    decoder = First_Decoder(W_s, C, f_c, nn.Sigmoid())
    # 1.5 sequential_layer
    sequential_decoder = Sequential_Decoder(W_s, C, nn.Sigmoid())
    # 1.6 decoder_layer
    decoder_layer = Decoder_layer(decoder, sequential_decoder, 2)
    # 1.7 decoder_encoder
    encoder_decoder = Encoder_Decoder(encoder_layer, decoder_layer)
    encoder_decoder.double()

