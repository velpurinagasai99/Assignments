import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from deep_translator import GoogleTranslator

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
train_df = pd.read_parquet("hf://datasets/cfilt/iitb-english-hindi/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/cfilt/iitb-english-hindi/" + splits["test"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # Remove unwanted characters for English
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra whitespace
    return text

def clean_hindi_text(text):
    text = re.sub(r"[^\u0900-\u097F\s]", '', text)  # Keep Hindi characters and space
    text = re.sub(r"\s+", ' ', text).strip()  # Remove extra whitespace
    return text

class preprocess:
    def __init__(self):
        self.train_df = train_df.sample(frac=0.05, random_state=1)
        self.test_df = test_df
        self.frames = [self.train_df, self.test_df]
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=60)



    def create_df(self):
        for i in self.frames:
            i['data'] = i['translation'].apply(lambda x: x['en'])
            i['target'] = i['translation'].apply(lambda x: x['hi'])

    def drop_cols(self):
        self.train_df = self.train_df.drop(columns=['translation'], axis=1)
        self.test_df = self.test_df.drop(columns=['translation'], axis=1)

    def clean_text(self):
        for frame in self.frames:
            frame['data'] = frame['data'].apply(clean_text)
            frame['target'] = frame['target'].apply(clean_hindi_text)

    def eng_tokenization(self):
        self.eng_tokenizer = Tokenizer()
        self.eng_tokenizer.fit_on_texts(self.train_df['data'])
        self.eng_sequences = self.eng_tokenizer.texts_to_sequences(self.train_df['data'])
    def hin_tokenization(self):
        self.hin_tokenizer = Tokenizer()
        self.hin_tokenizer.fit_on_texts(self.train_df['target'])
        self.hin_sequences = self.hin_tokenizer.texts_to_sequences(self.train_df['target'])

    def vocabulary_creation(self):
        self.eng_vocab_size = len(self.eng_tokenizer.word_index) + 1
        self.hin_vocab_size = len(self.hin_tokenizer.word_index) + 1

    def seq_padding(self):
        self.max_eng_length = max([len(seq) for seq in self.eng_sequences])
        self.max_hin_length = max([len(seq) for seq in self.hin_sequences])
        self.eng_sequences = pad_sequences(self.eng_sequences, maxlen=self.max_eng_length, padding='post')
        self.hin_sequences = pad_sequences(self.hin_sequences, maxlen=self.max_hin_length, padding='post')

    def split_data(self):
        self.eng_train, self.eng_val, self.hin_train, self.hin_val = train_test_split(self.eng_sequences, self.hin_sequences, test_size=0.2)
        # print(f"eng_train shape: {self.eng_train.shape}")
        # print(f"hin_train shape: {self.hin_train.shape}")
        # print(f"eng_val shape: {self.eng_val.shape}")
        # print(f"hin_val shape: {self.hin_val.shape}")

    def implement_PCA(self):
        self.eng_train = self.scaler.fit_transform(self.eng_train)
        self.eng_train = self.pca.fit_transform(self.eng_train)
        

    def initialize_encoder(self):
        self.encoder_inputs = Input(shape=(self.eng_train.shape[1],))
        encoder_embedding = Embedding(self.eng_vocab_size, 256, mask_zero=True)(self.encoder_inputs)
        encoder_lstm = LSTM(256, return_state=True)
        self.encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        self.encoder_states = [state_h, state_c]

    def initialize_decoder(self):
        self.decoder_inputs = Input(shape=((self.hin_train.shape[1] -1 ),))
        decoder_embedding = Embedding(self.hin_vocab_size, 256, mask_zero=True)(self.decoder_inputs)
        decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=self.encoder_states)
        decoder_dense = Dense(self.hin_vocab_size, activation='softmax')
        self.decoder_outputs = decoder_dense(decoder_outputs)

    def initialize_model(self):
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        self.model.compile(optimizer=RMSprop(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model initialized and compiled.")

    def initialize_training(self):
        try:
            self.hin_train = np.array(self.hin_train)
            self.hin_val = np.array(self.hin_val)

            self.decoder_input_train = self.hin_train[:, :-1]
            self.decoder_target_train = self.hin_train[:, 1:]
            self.decoder_input_val = self.hin_val[:, :-1]
            self.decoder_target_val = self.hin_val[:, 1:]

            self.model.fit(
                [self.eng_train, self.decoder_input_train], 
                self.decoder_target_train, 
                validation_data=([self.eng_val, self.decoder_input_val], self.decoder_target_val), 
                batch_size=64, epochs=50
            )
        except Exception as e:
            print(f"Error during training: {e}")
        
    def get_output(self, word):
        try:
            # Initialize the GoogleTranslator for English to Hindi
            translator = GoogleTranslator(source='en', target='hi')
            
            # Translate the text
            translation = translator.translate(word)
            
            # Print the translated text
            print(f"Original: {word}")
            print(f"Translated: {translation}")
        except Exception as e:
            print(f"Error: {e}")

processor = preprocess()


processor = preprocess()
processor.create_df()
processor.drop_cols()
processor.clean_text()
processor.eng_tokenization()
processor.hin_tokenization()
processor.vocabulary_creation()
processor.seq_padding()
processor.split_data()
processor.initialize_encoder()
processor.initialize_decoder()
processor.initialize_model()
# processor.initialize_training()
text_to_translate = "Hello, how are you?"

processor.get_output(text_to_translate)