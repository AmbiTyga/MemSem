from preprocessing import preprocess_image, preprocess_txt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image as keras_image
from metrics import precision, recall, f1

import numpy as np


class Classifier:

    def __init__(self, epochs=32, batch_size=64,metrics = False, plot_model_diagram=False, summary=False):
        self.epochs = epochs
        self.metrics = metrics
        self.batch_size = batch_size
        self.plot_model_diagram = plot_model_diagram
        self.summary = summary
        self.seq_len = 42
        self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vgg = VGG19(weights='imagenet', include_top=False)
        self.bert_layer.trainable = False
        self.vgg.trainable = False

    def encode(self, texts):
        input_id = []
        token_type_id = []
        attention_mask = []
        for text in texts:
            dictIn = self.tokenizer.encode_plus(text, max_length=self.seq_len, pad_to_max_length=True)
            input_id.append(dictIn['input_ids'])
            token_type_id.append(dictIn['token_type_ids'])
            attention_mask.append(dictIn['attention_mask'])
        return np.array(input_id), np.array(token_type_id), np.array(attention_mask)

    def labelencoder(self, labels):
        new_label = np.zeros((len(labels), 3))
        for i, label in enumerate(labels):
            if label == 0:
                new_label[i] = [0, 0, 1]
            elif label == 1:
                new_label[i] = [0, 1, 0]
            elif label == 2:
                new_label[i] = [1, 0, 0]

        return new_label

    def build(self):
        input_id = Input(shape=(self.seq_len,), dtype=tf.int64)
        mask_id = Input(shape=(self.seq_len,), dtype=tf.int64)
        seg_id = Input(shape=(self.seq_len,), dtype=tf.int64)

        _, bert_out = self.bert_layer([input_id, mask_id, seg_id])
        dense = Dense(768, activation='relu')(bert_out)
        dense = Dense(256, activation='relu')(dense)
        txt_repr = Dropout(0.4)(dense)
        ################################################
        img_in = Input(shape=(224, 224, 3))
        img_out = self.vgg(img_in)
        flat = Flatten()(img_out)
        dense = Dense(2742, activation='relu')(flat)
        dense = Dense(256, activation='relu')(dense)
        img_repr = Dropout(0.4)(dense)
        concat = Concatenate(axis=1)([img_repr, txt_repr])
        dense = Dense(64, activation='relu')(concat)
        out = Dense(3, activation='softmax')(dense)
        model = Model(inputs=[input_id, mask_id, seg_id, img_in], outputs=out)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(2e-5),
                      metrics=['accuracy', precision, recall, f1]) if self.metrics else model.compile(
            loss='categorical_crossentropy', optimizer=Adam(2e-5), metrics=['accuracy'])

        plot_model(model) if self.plot_model_diagram else None
        model.summary() if self.summary else None

        return model

    def train(self, data, validation_split=0.2):

        model = self.build()
        input_id, token_type_id, attention_mask = self.encode(data['text'])
        image_data = np.asarray(data['image'])
        labels = self.labelencoder(data['label'])

        self.history = model.fit([input_id, token_type_id, attention_mask, image_data],
                                 labels,
                                 validation_split=validation_split,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs)

        model.save_weights('./model/MemSem')

    def evaluate(self, data):
        model = self.build()
        model.load_weights('./model/MemSem')
        input_id, token_type_id, attention_mask = self.encode(data['text'].apply(preprocess_txt))
        image_data = data['image'].apply(preprocess_image)
        eval_data = [input_id, token_type_id, attention_mask,image_data]
        labels = self.labelencoder(data['label'])
        evaluation = model.evaluate(eval_data, labels)
        return evaluation


    def predict(self, image_path='./dataset/test/text.jpg', text=""):

        try:
            model = self.build()
            model.load_weights('./model/MemSem')
            input_id, token_type_id, attention_mask = self.encode([preprocess_txt(text)])
            image_data = preprocess_image(
                keras_image.load_img(image_path,
                                     target_size=(224, 224),
                                     interpolation='bicubic'))
            image_data = np.expand_dims(image_data, axis=0)
            value = model.predict([input_id, token_type_id, attention_mask, image_data])

            prediction = np.argmax(value)
            if prediction == 2:  # negative = [0,0,1]
                print("Its a bad meme")
            elif prediction == 1:  # postive = [0,1,0]
                print("Its not bad XD")
            elif prediction == 0:  # neutral = [1,0,0]
                print("Its meaningless")

        except Exception as e:
            print(e)
