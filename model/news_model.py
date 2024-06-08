import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attention_weights = torch.tanh(self.attention(lstm_output)).squeeze(2)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return weighted_output, attention_weights
    
class BiLSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm1(x, (h0, c0))

        # lstm_out, _ = self.lstm(cls_token_embedding.unsqueeze(1))
        attn_output, _ = self.attention(out)

        out = self.fc(attn_output)  # Take only the last timestep's output  [:, -1, :]
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out