import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3, bidirectional=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            bidirectional=bidirectional, dropout=dropout)
        
        self.direction_factor = 2 if bidirectional else 1
        self.layer_norm = nn.LayerNorm(hidden_size * self.direction_factor)
        self.attention = AttentionLayer(hidden_size * self.direction_factor)
        self.fc = nn.Linear(hidden_size * self.direction_factor, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.direction_factor, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.direction_factor, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.layer_norm(lstm_out)
        
        attention_out = self.attention(lstm_out)
        out = self.dropout(attention_out)
        out = self.fc(out)
        
        return out
