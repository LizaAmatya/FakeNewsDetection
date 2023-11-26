import os
import pandas as pd
import tensorflow as tf
import tensorflow_text
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.utils import to_categorical


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


labels = train['label']
label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
num_of_classes=len(label_mapping)
# print(num_of_classes)

train['label_encoded'] = train['label'].map(label_mapping)

val['label_encoded'] = val['label'].map(label_mapping)

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

# Custom embedding layer

# custom_embeddings = custom_embedding_layer(input_ids)

# Adding a dense layer for the output 
dense_layer = tf.keras.layers.Dense(num_of_classes, activation='softmax', name='dense_output')
bert_output = model([input_ids, attention_mask])
cls_token = bert_output.logits
dense_output = dense_layer(cls_token)
# dense_output_expanded = tf.keras.layers.Reshape((1, 6))(dense_output)

# combined_embeddings = tf.keras.layers.Concatenate(axis=-1)([dense_output_expanded, custom_embeddings])
output = tf.keras.layers.Dense(num_of_classes, activation='softmax')(dense_output)

# Create the final model
custom_model = tf.keras.Model(inputs=model.input, outputs=output)

custom_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),        #BinaryCrossEntropy for binary classification; for now lets only classify acc to data: 6 classes
            metrics=tf.keras.metrics.SparseCategoricalAccuracy())  # or use ['accuracy']

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
