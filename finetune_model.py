import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification
from common import BASE_DIR, train_dataset, val_dataset, num_epochs, batch_size
import os
from tensorflow.keras import mixed_precision


"""
BERT Preprocess Model
'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
"""
bert_encoders = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
}
mixed_precision.set_global_policy('mixed_float16')

bert_preprocess = {
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

# model_name = 'experts_wiki_books'
model_name = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = TFDistilBertForSequenceClassification.from_pretrained(model_name)

# print(model)
# bert_preprocess_model = hub.KerasLayer(preprocess)
# bert_model = hub.KerasLayer(encoder)

# text_test = ["I'm the only person on this stage who has worked actively just last year passing, along with Russ Feingold, some of the toughest ethics reform since Watergate.	ethics	barack-obama	President	Illinois	democrat	70.0	71.0	160.0	163.0	9.0	a Democratic debate in Philadelphia, Pa.	However, it was not that bill, but another one, sponsored by Majority Leader Harry Reid and introduced five days earlier on Jan.  4, 2007 that eventually became law. Obama was not a cosponsor"]
# text_preprocessed = bert_preprocess_model(text_test)

# bert_results = bert_model(text_preprocessed)

class BERT_classifier(object):
    
    def __init__(self, *args, **kwargs):
        # Using Expert Preprocessed BERT model
        self.encoder = bert_encoders['distilbert-base-uncased']
        self.preprocess = bert_preprocess['distilbert-base-uncased']
        
    def build_classifier_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)

classifier_model = BERT_classifier().build_classifier_model()
# bert_raw_result = classifier_model(tf.constant(text_test))
# print(tf.sigmoid(bert_raw_result))


history = classifier_model.fit(
    train_dataset.shuffle(10000).batch(batch_size),
    epochs=num_epochs,
    validation_data=val_dataset.batch(batch_size)
)

# Save the trained model
classifier_model.save_pretrained(os.path.join(BASE_DIR, 'trained_model'))
