# -*- coding: utf-8 -*-

base_file_name       = "Transformer-124"
train_file_name      = base_file_name+"-train.csv"
test_file_name       = base_file_name+"-test.csv"
validation_file_name = base_file_name+"-validation.csv"
model_file_name      = base_file_name+"-model.pt"

batch_size_start = 128#32  # original_value = 64
pos_weight = 1 # the value of pos_weight to resolve impalancing  original_value = .29
# learning rate
lr_v = 0.00007 # 1e-4 , 0.00007, 1e-5, 0.00005
n_epoch_number = 10
DROPOUT = 0.6 # 0.5 , 0.45
# reduced_dimensions = 768
random_s = 3

import torch

import pandas as pd
from torchtext.data import Field, BucketIterator, TabularDataset,LabelField
from sklearn.model_selection import train_test_split
import random
import numpy as np
from transformers import BertTokenizer
import torch.nn.functional as F

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import cohen_kappa_score

import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import LegalitySyllableTokenizer
from nltk import word_tokenize
# from nltk.tokenize import word_tokenize
import contractions
from nltk.corpus import words
from torchtext import data
LP = LegalitySyllableTokenizer(words.words())

from transformers import BertTokenizer, BertModel
# from transformers import BertTokenizer, BertModel

from transformers import BertTokenizer, BertModel
# from transformers import BertTokenizer, BertModel
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from textblob import TextBlob

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

import os
import logging

import numpy as np
from tqdm import trange
import tensorflow as tf

from utils import *
# from network import Network
# from statistic import Statistic

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

BERT_MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
len(tokenizer.vocab)

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(init_token, eos_token, pad_token, unk_token)

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

max_input_length = tokenizer.max_model_input_sizes[BERT_MODEL]
print(max_input_length)

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

from torchtext import data

Text = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx,
                  include_lengths = True)

Urgency = LabelField(sequential=False, use_vocab=False,dtype = torch.float)
fields = {"Text": ("t", Text), "Urgency": ("ur", Urgency)}

def sentimentAnalysis (document):

    vs = analyzer.polarity_scores(document)
#     res = TextBlob(document)
#     document = str(res)+ " " + document
# #     document = str(vs['neg'])+ " " + str(vs['pos'])+ " " +str(vs['compound'])+ " " + document
    if vs['compound'] >= 0.05:
        document = "0 " + document # "positive " + document
    elif vs['compound'] <= -0.05:
        document = "1 " + document # "negative " + document
    else:
        document = "1 " + document  #  "natural " + document

    return document

def preprocess (document):

    if "?" in document:
        document = "question " + document # document = "question " + document
    else:
        document = "answer " + document

    return document

# aug = nas.AbstSummAug(model_path='t5-base') # , num_beam=3

aug_1 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
aug_2 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
aug_3 = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
aug_4 = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute")
aug_5 = naw.SynonymAug(aug_src='wordnet')

def augment_text(df):
    new_text=[]
    ##selecting the minority class samples
    df_n=df[df.Urgency==1].reset_index(drop=True)
    ## data augmentation loop
    for i in range(len(df_n)):
#     for i in range(10):
#         print (i)
        text = df_n.loc[i, "Text"]
        augmented_text_1 = aug_1.augment(text)
        augmented_text_n_1 = "".join(augmented_text_1)
        tokens = tokenizer.tokenize(augmented_text_n_1)
        if (len(tokens)>=4):
            new_text.append(augmented_text_1)

        augmented_text_2 = aug_2.augment(text)
        augmented_text_n_2 = "".join(augmented_text_2)
        tokens = tokenizer.tokenize(augmented_text_n_2)
        if (len(tokens)>=4):
            new_text.append(augmented_text_2)

        augmented_text_3 = aug_3.augment(text)
        augmented_text_n_3 = "".join(augmented_text_3)
        tokens = tokenizer.tokenize(augmented_text_n_3)
        if (len(tokens)>=4):
            new_text.append(augmented_text_3)

        augmented_text_4 = aug_4.augment(text)
        augmented_text_n_4 = "".join(augmented_text_4)
        tokens = tokenizer.tokenize(augmented_text_n_4)
        if (len(tokens)>=4):
            new_text.append(augmented_text_4)

        augmented_text_5 = aug_5.augment(text)
        augmented_text_n_5 = "".join(augmented_text_5)
        tokens = tokenizer.tokenize(augmented_text_n_5)
        if (len(tokens)>=4):
            new_text.append(augmented_text_5)

    new=pd.DataFrame({'Text':new_text,'Urgency':1})
#     df=shuffle(df.append(new).reset_index(drop=True))
    df=df.append(new).reset_index(drop=True)
    return df


# GROUP A

df =  pd.read_excel('mystanfordMOOCForumPostsSet.xlsx')
df["Text"] = df["Text"].astype(str)
df["Text"] = df["Text"].apply(preprocess)
df["Text"] = df["CourseType"].astype(str)+" "+df["post_type"].astype(str)+" "+df["Text"].astype(str)
df["Text"] = df["Text"].apply(sentimentAnalysis)
df = df.loc[:, ['Text', 'Urgency']]
df.loc[df['Urgency']<4, ['Urgency']]=0
df.loc[df['Urgency']>=4, ['Urgency']]=1
print ('size of all data = ', df.shape)
train, test = train_test_split(df, test_size=0.2,stratify=df['Urgency'], random_state = random_s)
test,valid = train_test_split(test, test_size=0.5,stratify=test['Urgency'],random_state = random_s)
print ('size of all data = ', test.shape,valid.shape,train.shape)


train.to_csv("cnnTrain.csv", index=False) # cnnTrain_augmentation.csv
valid.to_csv("cnnValid.csv", index=False)
test.to_csv("cnnTest.csv", index=False)


train_data, valid_data, test_data = TabularDataset.splits(path = "", #path = 'D:\\ahmedFarouk\\proposal1_cnn\\'
                                                          train="cnnTrain_augmentation.csv",  # cnnTrain_augmentation.csv cnnTrain.csv
                                                          validation="cnnValid.csv",
                                                          test="cnnTest.csv", format="csv",
                                                          fields=fields)

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")


BATCH_SIZE = batch_size_start

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,sort_within_batch = True,
    sort_key = lambda x: len(x.t),
    device = device)


bert = AutoModel.from_pretrained(BERT_MODEL)

import torch.nn as nn

class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):

        super().__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
#         declar normalization for bert
        self.norm_embedding = nn.LayerNorm(embedding_dim)
#         declar Bi-LSTM
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
#         declar CNN - we have 4
        self.conv_0 = nn.Conv2d(in_channels = 1,
                                out_channels = n_filters,
                                kernel_size = (filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels = 1,
                                out_channels = n_filters,
                                kernel_size = (filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels = 1,
                                out_channels = n_filters,
                                kernel_size = (filter_sizes[2], embedding_dim))
        self.conv_3 = nn.Conv2d(in_channels = 1,
                                out_channels = n_filters,
                                kernel_size = (filter_sizes[3], embedding_dim))
#         declar self attention
        self.attention1 = torch.nn.MultiheadAttention(n_filters*4, 8, dropout=0.4)#
#         declar normalization after CNN
        self.norm1 = nn.LayerNorm(n_filters*4)
#         declar attention after LSTM
        self.attention4 = torch.nn.MultiheadAttention(hidden_dim*2, 8, dropout=0.4)#
#         declar normalization after LSTM
        self.norm4 = nn.LayerNorm(hidden_dim*2)

        self.attention5 = torch.nn.MultiheadAttention(n_filters*2+hidden_dim*2, 8, dropout=0.4)#
        self.norm5 = nn.LayerNorm(n_filters*2+hidden_dim*2)
        self.fc = nn.Linear(n_filters*2+hidden_dim*2, output_dim) # output_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
#         #text = [batch size, sent len]
        with torch.no_grad():
            embedded_all_layers = self.bert(text, output_hidden_states=True)[2]
#           embedded = self.bert(text)[0]
        embedded = embedded_all_layers[11]
#       normalzing the embedding layer
#         embedded = self.norm_embedding(embedded)
#         print('shape of embedding output ', embedded.shape)
        embedded_u = embedded.unsqueeze(1)

        embedded_p = embedded.permute(1, 0, 2)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded_p, text_lengths)
#         end of change 1
#         print ('packed_embedded = ', packed_embedded.)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
#         print ('hidden shape = ', hidden.shape)
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        conved_1 = F.relu(self.conv_2(embedded_u).squeeze(3))
        conved_2 = F.relu(self.conv_3(embedded_u).squeeze(3))
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        att1 = torch.cat((pooled_1, pooled_2), dim = 1)
        att4 = hidden
        temp = torch.cat((att1, att4), dim = 1)
        att5 = temp.unsqueeze(0)
        att5, _ = self.attention5(att5, att5, att5)
        att5= att5.squeeze(0)
        att5= self.norm5(att5)


        return self.fc(att5)

INPUT_DIM = len(tokenizer.vocab)#len(Text.vocab)
print(INPUT_DIM)
# EMBEDDING_DIM = 100
EMBEDDING_DIM = bert.config.to_dict()['hidden_size']
HIDDEN_DIM = 512 # the original value 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True

# PAD_IDX = Text.vocab.stoi[Text.pad_token]
n_filters = EMBEDDING_DIM
filter_sizes= [1, 2, 3, 4]
# filter_sizes= [2, 3, 4]

model = BERTGRUSentiment(bert,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

#for name, param in model.named_parameters():
    #if name.startswith('bert'):
       # param.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

from transformers import AdamW, get_linear_schedule_with_warmup
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr= lr_v) # 1e-4 , 0.00007, 1e-5, 0.00005
# optimizer = optim.Adam(model.parameters())
'''optimizer = AdamW(model.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )
#num_training_steps = N_EPOCHS+1

#scheduler = get_linear_schedule_with_warmup(optimizer,
                                                #num_warmup_steps=2, # Default value
                                                #num_training_steps=num_training_steps)'''

criterion = nn.BCEWithLogitsLoss()

# criterion.pos_weight = torch.ones([64])*8

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

#         predictions = model(batch.t).squeeze(1)
        text, text_lengths = batch.t
        predictions = model(text, text_lengths.cpu().numpy()).squeeze(1)

        criterion.pos_weight = torch.ones([int(batch.ur.shape[0])])*pos_weight
#         criterion.pos_weight.to (device)
        criterion = criterion.to(device)
#         print ("predictions shape = ", predictions.shape)
#         print ("batch-ur shape = ", batch.ur.shape)
        loss = criterion(predictions, batch.ur)

        acc = binary_accuracy(predictions, batch.ur)

        loss.backward()

        optimizer.step()
        #scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():

        for batch in iterator:

#             predictions = model(batch.t).squeeze(1)
            text, text_lengths = batch.t
            predictions = model(text, text_lengths.cpu().numpy()).squeeze(1)

            criterion.pos_weight = torch.ones([int(batch.ur.shape[0])])*pos_weight
#         criterion.pos_weight.to (device)
            criterion = criterion.to(device)

            loss = criterion(predictions, batch.ur)

            acc = binary_accuracy(predictions, batch.ur)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            #predictions = map(myfunc, predictions)

            predictions = torch.round(torch.sigmoid(predictions)).float()

            target = batch.ur.cpu().numpy()

            y_true.extend(predictions.tolist())
            y_pred.extend(target)
#             y_pred.extend(predictions.tolist())
#             y_true.extend(target)


    return epoch_loss / len(iterator), epoch_acc / len(iterator),y_true,y_pred

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def reduce_lr():
    print("Reducing LR")
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] *0.75
#         g['lr'] = g['lr'] *0.73

import matplotlib.pyplot as plt

def my_plot(epochs, loss):
    plt.plot(epochs, loss)
    plt.title('Model Loss (BERT)')
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.legend(["Train",'Validation'],loc='upper right')

def my_plotACC(epochs, acc):
    plt.plot(epochs, acc)
    plt.title('Model accuracy (BERT)')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')
    plt.legend(["Train",'Validation'],loc='upper left')
N_EPOCHS = n_epoch_number

best_valid_loss = float('inf')
best_valid_acc = float('-inf')
train_loss_vals=  []
valid_loss_vals=  []

train_acc_vals=  []
valid_accs_vals=  []

for epoch in range(N_EPOCHS):
    #if (epoch > 0) and (epoch % 10 == 0):
           #reduce_lr()

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc,_,_  = evaluate(model, valid_iterator, criterion)

    train_loss_vals.append(train_loss)
    valid_loss_vals.append(valid_loss)
    train_acc_vals.append(train_acc)
    valid_accs_vals.append(valid_acc)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'tut6-model.pt')
        torch.save(model.state_dict(), model_file_name)
        best_epoch = epoch+1

    '''if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'tut6-model.pt')
        best_epoch = epoch+1 '''

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    #     new lines i addes to monitor test results
    print ('test results for this epoch = ', epoch+1)
    test_loss, test_acc, y_true,y_pred = evaluate(model, test_iterator, criterion)
    print(classification_report(y_true, y_pred,digits = 3))
    print(confusion_matrix(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    print('Weighted_F1',f1_score(y_true, y_pred, average='weighted'))
    print('cohen_kappa_score',cohen_kappa_score(y_true, y_pred))

    lr_precision, lr_recall, _ = precision_recall_curve(y_true,y_pred)
    print('lr_precision, lr_recall',lr_precision, lr_recall)
    lr_f1, lr_auc = f1_score(y_true,y_pred), auc(lr_recall, lr_precision)

    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    ones = [x for x in y_true if x==1]
    print('Length',len(ones),len(y_true))
    no_skill = len(ones) / len(y_true)
    print('  no_skill ', no_skill)
    print('                                  ')
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='BERT')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
# end of update
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.3f}%')

print('Best epoch= ',best_epoch)
#my_plotACC(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), train_acc_vals)
#my_plotACC(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), valid_accs_vals)
my_plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), train_loss_vals)
my_plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), valid_loss_vals)


model.load_state_dict(torch.load(model_file_name))
test_loss, test_acc, y_true,y_pred = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.3f}%')
print(classification_report(y_true, y_pred,digits = 3))

print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
print('Weighted_F1',f1_score(y_true, y_pred, average='weighted'))
print('cohen_kappa_score',cohen_kappa_score(y_true, y_pred))

lr_precision, lr_recall, _ = precision_recall_curve(y_true,y_pred)
print('lr_precision, lr_recall',lr_precision, lr_recall)
lr_f1, lr_auc = f1_score(y_true,y_pred), auc(lr_recall, lr_precision)

print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
ones = [x for x in y_true if x==1]
print('Length',len(ones),len(y_true))
no_skill = len(ones) / len(y_true)
print('  no_skill ', no_skill)
print('                                  ')
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='BERT')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

