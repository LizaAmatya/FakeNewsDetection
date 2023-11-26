import os
import pandas as pd
import tensorflow as tf
import tensorflow_text
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.utils import to_categorical
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["TF_AUTOTUNE"] = "1"
tf.keras.backend.set_floatx('float16')

# Enable autotuning
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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

# Mapping to binary classes

labels = train['label']
label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
print('label mapping', label_mapping)

train['label_encoded'] = np.where(np.isin(labels, ['mostly-true', 'true']), 1, 0)   # Mapping True as 1 and lies as 0
val['label_encoded'] = np.where(np.isin(val['label'], ['mostly-true', 'true']), 1, 0)

num_of_classes=2

print(train['label_encoded'])

# train_one_hot_labels = to_categorical(train['label_encoded'], num_classes=num_of_classes)
# val_one_hot_labels = to_categorical(val['label_encoded'], num_classes=num_of_classes)

# model_name = 'experts_wiki_books'
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
vocab_size = 10000  
embedding_dim = 32  

# Create a custom embedding layer
custom_embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

# metadata_columns = ['subjects', 'speakers', 'jobs', 'states', 'affiliations', 'contexts']

# Only using statement data at first
# Tokenize the statement data
train_encoded_statement_data = tokenizer(
    train['statement'].to_list(),
    padding=True,
    truncation=True,
    return_tensors='tf'
)

# train_labels = train['label'].tolist()

val_encoded_statement_data = tokenizer(
    val['statement'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='tf'
)

# val_labels = val['label'].tolist()

# Create TensorFlow dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': train_encoded_statement_data['input_ids'], 
        'attention_mask': train_encoded_statement_data['attention_mask']
    }, 
    train['label_encoded'] ))  # using one-hot encoded labels when CategoricalCrossEntropy used, 
                            # and when using SparseCrossEntropy use train['label_encoded'] which is int rep for labels : 0, 1, 2 ..5

val_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': val_encoded_statement_data['input_ids'], 
        'attention_mask': val_encoded_statement_data['attention_mask']
    },
    val['label_encoded'] ))  # using one-hot encoded labels when CategoricalCrossEntropy used, 
                            # and when using SparseCrossEntropy use train['label_encoded'] which is int rep for labels : 0, 1, 2 ..5

# Limiting the dataset
limit = 100  
limited_train_dataset = train_dataset.take(limit)

model = TFBertForSequenceClassification.from_pretrained(model_name)

num_epochs = 4
batch_size = 16

# # Create a new model with the BERT base and the custom output layer
input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

# Adding a dense layer for the output 
# dense_layer = tf.keras.layers.Dense(num_of_classes, activation='softmax', name='dense_output')
bert_output = model([input_ids, attention_mask])
cls_token = bert_output.logits
positive_class_logits = cls_token[:, 1]  # Extract logits for the positive class 
positive_class_logits = tf.expand_dims(positive_class_logits, axis=-1)

dense_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_output')
output = dense_layer(positive_class_logits)

output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

# Create the final model
custom_model = tf.keras.Model(inputs=model.input, outputs=output)

custom_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy'])

custom_model.summary()

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(BASE_DIR, 'model_checkpoint'),  # Specify the path to save the checkpoint
    save_best_only=True,  # Save only the best model based on the validation loss
    monitor='val_loss',  # Monitor the validation loss
    mode='min',  # Mode can be 'min' or 'max' depending on the monitored metric
    verbose=1  # Show progress while saving
)

print('Start training')
history = custom_model.fit(
    limited_train_dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE),
    epochs=num_epochs,
    validation_data=val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
    verbose=1,
    callbacks=[checkpoint_callback]
)

# Save the trained model if needed
custom_model.save_pretrained(os.path.join(BASE_DIR, 'trained_model'))
