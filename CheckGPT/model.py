import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
# from ptflops import get_model_complexity_info
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaPreTrainedModel


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False, device="cuda"):
        super(Attention, self).__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        stdv = 1.0 / np.sqrt(self.hidden_size)
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.device = device

        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)
        mask = torch.ones(attentions.size(), requires_grad=True)
        if self.device == "cuda":
            mask = mask.cuda()
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)
        attentions = masked.div(_sums)
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        return weighted.sum(1), attentions


class CheckGPT(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, batch_first=True, dropout=0.5, bidirectional=True,
                 num_layers=2, device="cuda", v1=0):
        super(CheckGPT, self).__init__()
        self.lstm = nn.GRU if not v1 else nn.LSTM
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.lstm1 = self.lstm(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=bidirectional,
                          batch_first=batch_first)
        self.atten1 = Attention(hidden_size * 2, batch_first=batch_first, device=device)
        self.lstm2 = self.lstm(input_size=hidden_size * 2,
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=bidirectional,
                          batch_first=batch_first)
        self.atten2 = Attention(hidden_size * 2, batch_first=batch_first, device=device)
        self.fc = nn.Linear(hidden_size * num_layers * 2, 2)

    def forward(self, x0, lengths):
        packed_input = pack_padded_sequence(x0, lengths, batch_first=self.batch_first, enforce_sorted=False)
        out1, (_, _) = self.lstm1(packed_input)
        out1, _ = pad_packed_sequence(out1, batch_first=self.batch_first)
        x, _ = self.atten1(out1)

        packed_input2 = pack_padded_sequence(out1, lengths, batch_first=self.batch_first, enforce_sorted=False)
        out2, (_, _) = self.lstm2(packed_input2)
        out2, _ = pad_packed_sequence(out2, batch_first=self.batch_first)
        y, _ = self.atten2(out2)

        z = torch.cat([x, y], dim=1)
        z = self.fc(self.dropout(z))
        return z


class RobertaClassificationHead(nn.Module):
    def __init__(self):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(1024, 2)

    def forward(self, input, length):
        x = input[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaMeanPoolingClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(1024, 1024)
        self.out_proj = nn.Linear(1024, 2)

    def forward(self, features, non_padding_token_count, **kwargs):
        attentions_mask = torch.stack([
            torch.cat([torch.ones(non_padding_token_count[i].item()), torch.zeros(features.size(1) - non_padding_token_count[i].item())])
            for i in range(len(non_padding_token_count))
        ]).to(features.device)
        masked_features = features * attentions_mask.unsqueeze(-1)
        summed_features = torch.sum(masked_features, dim=1)
        mean_pooled_features = summed_features / non_padding_token_count.unsqueeze(-1).to(features.device)
        x = self.dropout(mean_pooled_features)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_labels=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('f_conv1', nn.Conv2d(1, 32, kernel_size=11, stride=(2, 3), padding=(5, 5)))

        self.feature1 = nn.Sequential()
        self.feature1.add_module('f_bn1', nn.BatchNorm2d(32))
        self.feature1.add_module('f_pool1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.feature1.add_module('f_relu1', nn.ReLU(True))

        self.feature2 = nn.Sequential()
        self.feature2.add_module('f_conv2', nn.Conv2d(32, 64, kernel_size=(3, 5), stride=2, padding=(2, 2)))
        self.feature2.add_module('f_conv3', nn.Conv2d(64, 64, kernel_size=(3, 5)))
        self.feature2.add_module('f_bn2', nn.BatchNorm2d(64))
        self.feature2.add_module('f_drop1', nn.Dropout2d())
        self.feature2.add_module('f_pool2', nn.MaxPool2d(kernel_size=3, stride=2))
        self.feature2.add_module('f_relu2', nn.ReLU(True))

        self.feature3 = nn.Sequential()
        self.feature3.add_module('f_conv4', nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 1)))
        self.feature3.add_module('f_conv5', nn.Conv2d(128, 128, kernel_size=(3, 5)))
        self.feature3.add_module('f_bn3', nn.BatchNorm2d(128))
        self.feature3.add_module('f_drop3', nn.Dropout2d())
        self.feature3.add_module('f_pool3', nn.MaxPool2d(kernel_size=3, stride=2))
        self.feature3.add_module('f_relu3', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(128 * 14 * 16, 100))
        self.class_classifier.add_module('c_drop1', nn.Dropout1d())
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, num_labels))

    def forward(self, input_data, lengths):
        data_after_conv1 = self.conv1(input_data)
        feature1 = self.feature1(data_after_conv1)
        feature2 = self.feature2(feature1)
        feature = self.feature3(feature2)

        size_ = feature.shape[-1] * feature.shape[-2] * feature.shape[-3]
        feature = feature.view(-1, size_)
        class_output = self.class_classifier(feature)
        return class_output


class LSTMwoAttention(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, batch_first=True, dropout=0.5, bidirectional=True,
                 num_layers=2, device="cuda", v1=0):
        super(LSTMwoAttention, self).__init__()
        self.lstm = nn.LSTM if not v1 else nn.GRU
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.rnn = self.lstm(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first)
        self.fc = nn.Linear(hidden_size * num_layers, 2)

    def forward(self, x0, lengths):
        packed_input = pack_padded_sequence(x0, lengths, batch_first=self.batch_first, enforce_sorted=False)
        out1, _ = self.rnn(packed_input)
        x, _ = pad_packed_sequence(out1, batch_first=self.batch_first)
        select_x = x[torch.arange(len(lengths)), lengths - 1]
        z = self.fc(self.dropout(select_x))
        return z


if __name__ == '__main__':
    pass
    # config = RobertaConfig.from_pretrained("roberta-large", num_labels=2)
    # m = RobertaModel(config).cuda()
    # print(get_model_complexity_info(m, (512,), as_strings=True, print_per_layer_stat=True))