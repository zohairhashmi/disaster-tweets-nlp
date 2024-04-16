from torch import nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name) # BERT model - pre-trained model from Hugging Face Transformers
        self.dropout = nn.Dropout(0.2) # dropout layer - randomly zeroes some of the elements of the input tensor with probability p
        # self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 384) # fully connected layer - applies a linear transformation to the incoming data
        self.fc2 = nn.Linear(384, num_classes) # fully connected layer - applies a linear transformation to the incoming data

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # BERT model forward pass - returns a tuple of last hidden states, pooler output, hidden states, and attentions
        pooled_output = outputs.pooler_output # pooled output - output of the model's classifier (hidden state of the first token of the sequence)
        # pooled_output = self.norm(pooled_output) # layer normalization - normalizes the activations of the previous layer for each data point
        x = self.dropout(pooled_output) # dropout layer forward pass - randomly zeroes some of the elements of the input tensor with probability p
        x = nn.ReLU()(self.fc1(x))
        logits = self.fc2(x)
        return logits