import os
import pandas as pd
import tensorflow as tf
import tensorflow_text
from transformers import BertTokenizer, TFBertForSequenceClassification


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, 'dataset/train.tsv')
test_path = os.path.join(BASE_DIR, 'dataset/test.tsv')
validation_path = os.path.join(BASE_DIR, 'dataset/validation.tsv')

column_labels = ['row', 'json_ids', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state', 'affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'lies_counts', 'context', 'justification']

# Data Frames 
train = pd.read_csv(train_path, sep="\t", header=None, names=column_labels)
test = pd.read_csv(test_path, sep="\t", header=None, names=column_labels)
valid = pd.read_csv(validation_path, sep="\t", header=None, names=column_labels)

# Fill nan (empty boxes) with 0
train = train.fillna('None')
test = test.fillna('None')
val = valid.fillna('None')

# train = train.values
# test = test.values
# val = val.values

# model_name = 'experts_wiki_books'
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

train_encoded_statement_data = tokenizer(
    train['statement'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='tf'
)

train_labels = train['label'].tolist()

val_encoded_statement_data = tokenizer(
    val['statement'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='tf'
)

val_labels = val['label'].tolist()

# Create TensorFlow dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': train_encoded_statement_data['input_ids'], 'attention_mask': train_encoded_statement_data['attention_mask']}, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': val_encoded_statement_data['input_ids'], 'attention_mask': val_encoded_statement_data['attention_mask']}, train_labels))

# model = TFBertForSequenceClassification.from_pretrained(model_name)
num_epochs = 10
batch_size = 32
