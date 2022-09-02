# ref: https://github.com/killerray/dnnBox

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models.convnext import LayerNorm2d

class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, bidirectional):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True,
                            bidirectional=bidirectional)
        if self.bidirectional:
            self.classifier = nn.Linear(hidden_dim * 2, n_class)
        else:
            self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):

        out, (hn, _) = self.lstm(x)
        if self.bidirectional:
            out = torch.hstack((hn[-2, :, :], hn[-1, :, :]))
        else:
            out = out[:, -1, :]
        out = self.classifier(out)
        return out


class Attention(nn.Module):
    def __init__(self, rnn_size: int):
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        # eq.9: M = tanh(H)
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)

        # eq.10: α = softmax(w^T M)
        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)

        # eq.11: r = H
        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        r = r.sum(dim=1)  # (batch_size, rnn_size)

        return r, alpha


class AttBiLSTM(nn.Module):
    def __init__(
            self,
            n_classes: int,
            emb_size: int,
            rnn_size: int,
            rnn_layers: int,
            dropout: float
    ):
        super(AttBiLSTM, self).__init__()

        self.rnn_size = rnn_size

        # bidirectional LSTM
        self.BiLSTM = nn.LSTM(
            emb_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        self.attention = Attention(rnn_size)
        self.fc = nn.Linear(rnn_size, n_classes)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        rnn_out, _ = self.BiLSTM(x)

        H = rnn_out[:, :, : self.rnn_size] + rnn_out[:, :, self.rnn_size:]

        # attention module
        r, alphas = self.attention(
            H)  # (batch_size, rnn_size), (batch_size, word_pad_len)

        # eq.12: h* = tanh(r)
        h = self.tanh(r)  # (batch_size, rnn_size)

        scores = self.fc(self.dropout(h))  # (batch_size, n_classes)

        return scores


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Linear(32 * 5 * 5, 120),
                                        nn.Linear(120, 84),
                                        nn.Linear(84, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.classifier(x)
        return x


class LeNetVariant(nn.Module):
    def __init__(self):
        super(LeNetVariant, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Linear(32 * 5 * 5, 120),
                                        nn.Linear(120, 84))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.classifier(x)
        return x


def build_action_encoder(config):
    encoder_type = config['action_encoder']
    dim_in = config['num_cls']
    if encoder_type == 'convnext_small':
        encoder = torchvision.models.convnext_small()
        encoder._modules['features'][0][0] = nn.Conv2d(dim_in, 96, kernel_size=(4, 4), stride=(4, 4))
        encoder.classifier = torch.nn.Sequential(LayerNorm2d(config['encoder_cout']),
                                                 torch.nn.Flatten())
    elif encoder_type == 'efficientnet_v2_s':
        encoder = torchvision.models.efficientnet_v2_s()
        encoder.features[0][0] = torch.nn.Conv2d(dim_in, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        encoder.classifier = torch.nn.Dropout(p=0.2, inplace=True)
    elif encoder_type == 'resnet18':
        # replace first layer with segmentation mask dim
        encoder = torchvision.models.resnet18()
        encoder.conv1 = torch.nn.Conv2d(dim_in, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
        encoder.fc = torch.nn.Identity()  # output feature vector only
    elif encoder_type == 'resnet50':
        encoder = torchvision.models.resnet50()
        encoder.conv1 = torch.nn.Conv2d(dim_in, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
        encoder.fc = torch.nn.Identity()  # output feature vector only
    elif encoder_type == "mnasnet0_5":
        encoder = torchvision.models.mnasnet0_5()
        encoder.layers[0] = torch.nn.Conv2d(10, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        encoder.classifier = torch.nn.Identity()
    else:
        raise ValueError('Not implemented encoder_type')
    return encoder

class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.encoder(x)[-1]  # encdoer output a list of different stages , take last stage's output
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, config, encoder=None, num_classes=8):
        super(CNNLSTM, self).__init__()
        encoder_cout = config['encoder_cout']
        if encoder:
            self.cnn = encoder  # EncoderWrapper(encoder)
        else:
            raise ValueError('Not implemented yet')

            # self.cnn = LeNetVariant()
            # cnn = torchvision.models.mobilenet_v3_small()  # 576
            # cnn.classifier = nn.Identity()  # output feature vector only

            # replace first layer with segmentation mask dim
            encoder = torchvision.models.resnet18()
            encoder.conv1 = torch.nn.Conv2d(config['num_cls'], 64, kernel_size=7, stride=2, padding=3,
                                            bias=False)
            encoder.fc = torch.nn.Identity()  # output feature vector only
            # print(encoder)
            self.cnn = encoder

        self.lstm = nn.LSTM(input_size=encoder_cout, hidden_size=128, num_layers=2,
                            batch_first=True)
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x_3d, state):
        # input shape: (seq_len, c, h, w)
        # print(x_3d.shape)

        x = self.cnn(x_3d).unsqueeze(0)  # (1, seq_len, c')

        # cnn_output_list = list()
        # for t in range(x_3d.size(0)):
        #     frame = x_3d[t, :, :, :]
        #     cnn_output = self.cnn(frame.unsqueeze(0))
        #     # print('cnn_output ', cnn_output.shape)
        #     cnn_output_list.append(cnn_output)
        # x = torch.stack(tuple(cnn_output_list), dim=1)  # (1, seq_len, c)
        # print('x ', x.shape)

        if not state:
            out, (h, c) = self.lstm(x)
        else:
            out, (h, c) = self.lstm(x, state)
        # print('out ', out.shape)
        # h = h.permute(1, 0, 2)  # (D*num_layers, c_out)
        # c = c.permute(1, 0, 2)  # (D*num_layers, c_out)

        x = out[:, :, :]  # (bs, seq_len, num_cls)
        x = F.relu(x)
        logits = self.fc1(x).squeeze(0)  # remove batch dim
        # print('logits ', logits.shape)

        return logits, (h, c)


if __name__ == '__main__':
    print('gg')
    # inputs = torch.normal(0, 1, (1, 3, 224, 224))
    inputs = torch.normal(0, 1, (1, 1, 3, 224, 224))
    # cnn = torchvision.models.mobilenet_v3_small()
    # cnn.classifier = nn.Identity()
    # outputs = cnn(inputs) # 576

    model = CNNLSTM()
    outputs, (h, c) = model(inputs, None)
    h_temp, c_temp = h, c
    outputs, (h, c) = model(inputs, (h_temp, c_temp))
    print(outputs.shape, h.shape, c.shape)
