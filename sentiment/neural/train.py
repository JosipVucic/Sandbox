# data manipulation
import pandas as pd
import numpy as np

# pytorch
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchtext.vocab import build_vocab_from_iterator

# utils
from tqdm import tqdm
tqdm.pandas()

from sentiment.neural.model import SentimentModel
from sentiment.neural.preprocessing import pad_features


def get_vocab():
    data = pd.read_csv('./data/imdb_processed.csv')
    reviews = data.processed.values
    words = ' '.join(reviews)
    words = words.split()
    vocab = build_vocab_from_iterator([words], specials=["<PAD>"])

    # encode words and get labels
    reviews_enc = [vocab.forward(review.split()) for review in tqdm(reviews)]
    features = pad_features(reviews_enc, pad_id=vocab.get_stoi()['<PAD>'])

    # get labels as numpy
    labels = data.label.to_numpy()

    return features, labels, vocab


def get_model(vocab):
    # model hyperparamters
    vocab_size = len(vocab.get_stoi())
    output_size = 1
    embedding_size = 256
    hidden_size = 512
    n_layers = 2
    dropout = 0.25

    # model initialization
    return SentimentModel(vocab, vocab_size, output_size, hidden_size, embedding_size, n_layers, dropout)


def get_loaders(features, labels):
    # create tensor datasets
    baseset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    trainset, testset = random_split(baseset, [0.7, 0.3])
    testset, validset = random_split(testset, [0.5, 0.5])

    # define batch size
    batch_size = 128

    # create dataloaders
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(validset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    return trainloader, valloader, testloader


def train():
    features, labels, vocab = get_vocab()
    model = get_model(vocab)
    trainloader, valloader, testloader = get_loaders(features, labels)

    # training config
    lr = 0.001
    criterion = nn.BCELoss()  # we use BCELoss cz we have binary classification problem
    optim = Adam(model.parameters(), lr=lr)
    grad_clip = 5
    epochs = 8
    print_every = 1
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': epochs
    }
    es_limit = 5

    # define training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)

    #################
    # TRAIN LOOP #
    #################

    epochloop = tqdm(range(epochs), position=0, desc='Training', leave=True)

    # early stop trigger
    es_trigger = 0
    val_loss_min = torch.inf

    for e in epochloop:

        #################
        # training mode #
        #################

        model.train()

        train_loss = 0
        train_acc = 0

        for id, (feature, target) in enumerate(trainloader):
            # add epoch meta info
            epochloop.set_postfix_str(f'Training batch {id}/{len(trainloader)}')

            # move to device
            feature, target = feature.to(device), target.to(device)

            # reset optimizer
            optim.zero_grad()

            # forward pass
            out = model(feature)

            # acc
            predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
            equals = predicted == target
            acc = torch.mean(equals.type(torch.FloatTensor))
            train_acc += acc.item()

            # loss
            loss = criterion(out.squeeze(), target.float())
            train_loss += loss.item()
            loss.backward()

            # clip grad
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # update optimizer
            optim.step()

            # free some memory
            del feature, target, predicted

        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc / len(trainloader))

        ####################
        # validation mode #
        ####################

        model.eval()

        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for id, (feature, target) in enumerate(valloader):
                # add epoch meta info
                epochloop.set_postfix_str(f'Validation batch {id}/{len(valloader)}')

                # move to device
                feature, target = feature.to(device), target.to(device)

                # forward pass
                out = model(feature)

                # acc
                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
                equals = predicted == target
                acc = torch.mean(equals.type(torch.FloatTensor))
                val_acc += acc.item()

                # loss
                loss = criterion(out.squeeze(), target.float())
                val_loss += loss.item()

                # free some memory
                del feature, target, predicted

            history['val_loss'].append(val_loss / len(valloader))
            history['val_acc'].append(val_acc / len(valloader))

        # reset model mode
        model.train()

        # add epoch meta info
        epochloop.set_postfix_str(
            f'Val Loss: {val_loss / len(valloader):.3f} | Val Acc: {val_acc / len(valloader):.3f}')

        # print epoch
        if (e + 1) % print_every == 0:
            epochloop.write(
                f'Epoch {e + 1}/{epochs} | Train Loss: {train_loss / len(trainloader):.3f} Train Acc: {train_acc / len(trainloader):.3f} | Val Loss: {val_loss / len(valloader):.3f} Val Acc: {val_acc / len(valloader):.3f}')
            epochloop.update()

        # save model if validation loss decrease
        if val_loss / len(valloader) <= val_loss_min:
            torch.save(model.state_dict(), './sentiment_lstm.pt')
            val_loss_min = val_loss / len(valloader)
            es_trigger = 0
        else:
            epochloop.write(
                f'[WARNING] Validation loss did not improved ({val_loss_min:.3f} --> {val_loss / len(valloader):.3f})')
            es_trigger += 1

        # force early stop
        if es_trigger >= es_limit:
            epochloop.write(f'Early stopped at Epoch-{e + 1}')
            # update epochs history
            history['epochs'] = e + 1
            break


if __name__ == "__main__":
    #train()

    _, _, vocab = get_vocab()

    torch.save(vocab, "./data/vocab.pth")