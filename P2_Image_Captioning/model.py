import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.5):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
        #                    dropout=drop_prob, batch_first=True) 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        
        #self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions): 
        
        batch_size = features.shape[0] 
        self.hidden = (torch.zeros(1, batch_size, self.hidden_size).to(self.device), torch.zeros(1, batch_size, self.hidden_size).to(self.device))
        
        embedding = self.embedding(captions[:, :-1])
        embedding = torch.cat((features.unsqueeze(1), embedding), dim=1)         
        lstm_out, self.hidden = self.lstm(embedding, self.hidden) 
        
        #out = self.dropout(lstm_out)
        
        out = self.fc(lstm_out)
        
        return out
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        #hidden = (torch.zeros(1, inputs.shape[0], self.hidden_size).to(self.device), torch.zeros(1, inputs.shape[0], self.hidden_size).to(self.device))
      
        out_list = []
        for i in range(0, max_len):
            lstm_out, states = self.lstm(inputs, states)
            _ , out = self.fc(lstm_out.squeeze(1)).max(dim=1)
            #print(out)
            out_list.append(out.item())
            inputs = self.embedding(out).unsqueeze(1)
            if out.item() == 1:
                break
        return out_list    