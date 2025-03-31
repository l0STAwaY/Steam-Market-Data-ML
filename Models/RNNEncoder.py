
import torch 
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, vocabSize, nEmbed, nHidden, nLayers, nClass,dropout=0.25):
        super(RNNEncoder, self).__init__()

        
        self.vocabSize = vocabSize 
        self.nEmbed = nEmbed  # the number of words I want to embed  
        self.nHidden = nHidden
        self.nLayers = nLayers
        self.nClass = nClass 
        
        # embed only takes int
        # but we still want float as out put bcause gradient accept only float or else
        # Only Tensors of floating point and complex dtype can require gradients
        self.Embedding_layer = nn.Embedding(self.vocabSize, self.nEmbed,dtype = torch.float) 
        
        # This definition creates the expected hidden size
        # the embed dimention equals the  
        # Embed dimention is actual (Batch, input, hidden) so we need bath first input
        # Input size seems to be for each time stamp 
        self.RNN_Layer = nn.RNN(input_size = self.nEmbed, hidden_size = self.nHidden, num_layers = self.nLayers , batch_first = True)
    
        self.Linear_Layer = nn.Linear(self.nHidden, self.nClass) # whicih word is next
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()
        
        

    def forward(self, X, hidden):
        ## Forward pass returns prediction and hidden
        # print("Input X:", X)  # Print the raw input tensor
        # print("Shape of X:", X.shape)  # Print the shape of the input tensor
        # if X.min() < 0 or X.max() >= self.vocabSize:
        #    raise ValueError(f"Indices out of bounds: min={X.min()}, max={X.max()}")
        
        Embeded_X = self.Embedding_layer(X)
        
        
        # print("Embeded X:", Embeded_X.shape)
        # print("Shape of hidden:", hidden.shape)
        
        
        output, hidden = self.RNN_Layer(Embeded_X,hidden)
        
        # if output is None:
        #    print("RNN output is None!")
        # print("Shape of output :", output.shape)

         # n batch, last time step all hidden size (the hidden size still needs softmax)
        
        # batch * timestep * hidden_features sumption
        
        final_output = self.Linear_Layer(self.dropout(output))
        
        final_output  = final_output[:, -1, :]

        sig_out = self.sig(final_output)

        return sig_out, hidden
 
    
    def init_hidden(self, batchSize):
        return torch.zeros((self.nLayers, batchSize, self.nHidden), dtype=torch.float)
      
        

    def loss(self, y_hat, y):
        ## Using pytorch cross-entropy loss
        # https://neptune.ai/blog/pytorch-loss-functions#:~:text=The%20Pytorch%20Cross%2DEntropy%20Loss,default%20loss%20function%20in%20Pytorch.
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(y_hat, y)
     