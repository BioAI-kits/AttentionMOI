from .Autoencoder import *
from .ClassificationModel import *

class Moanna(nn.Module):
    def __init__(self, input_shape, hidden_size, encoded_size, n_layers, drop_prob, fnn_hidden_size, num_classes, fnn_n_layers, fnn_p):
        super(Moanna, self).__init__()
        self.encoder = Encoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        self.decoder = Decoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        self.classifier = RLModel(encoded_size, fnn_hidden_size, num_classes, fnn_n_layers, fnn_p)
        
    def forward(self, x):
        outs = [] 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # out = self.classifier(encoded)
        
        return encoded, decoded
    
class Moanna_cls(nn.Module):
    def __init__(self, input_shape, hidden_size, encoded_size, n_layers, drop_prob, fnn_hidden_size, num_classes, fnn_n_layers, fnn_p):
        super(Moanna_cls, self).__init__()
        self.encoder = Encoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        self.decoder = Decoder(input_shape, hidden_size, encoded_size, n_layers, drop_prob)
        self.classifier = RLModel(encoded_size, fnn_hidden_size, num_classes, fnn_n_layers, fnn_p)
        
    def forward(self, encoded):
        # outs = [] 
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        out = self.classifier(encoded)
        
        return out