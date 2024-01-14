import torch
import torch.nn as nn


class SentimentModel(nn.Module):

    def __init__(self, vocab, vocab_size, output_size, hidden_size=128, embedding_size=400, n_layers=2, dropout=0.2):
        super(SentimentModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        self.vocab = vocab

    def forward(self, x):
        x = self.embedding(x)

        # get the hidden state outputs for all time steps, the cell states are in _
        o, _ = self.lstm(x)

        # get last sequence output
        o = o[:, -1, :]

        o = self.dropout(o)
        o = self.fc(o)

        # sigmoid
        o = self.sigmoid(o)

        return o

    def classify_input(self, review):
        """Returns sentiment classification for the provided input.
        :param review: Preprocessed review, list of tokens.
        :return: integer, one of the digits 0 to 9"""
        with torch.no_grad():
            tensor = torch.tensor(review)
            output = self(tensor).squeeze(0)

            predicted = "POSITIVE" if output.data.item() > 0.5 else "NEGATIVE"

        return predicted

    @classmethod
    def get_trained(cls):
        """Retrieves a pretrained model from ./sentiment/neural/trained_models/default.pth
        :return: The loaded Sentiment model."""

        # model hyperparamters
        vocab = torch.load("sentiment/neural/data/vocab.pth")
        vocab_size = len(vocab.get_stoi())
        output_size = 1
        embedding_size = 256
        hidden_size = 512
        n_layers = 2
        dropout = 0.25

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = SentimentModel(vocab, vocab_size, output_size, hidden_size, embedding_size, n_layers, dropout)
        model.load_state_dict(torch.load("sentiment/neural/trained_models/default.pth", map_location=device))
        model.eval()

        return model
