# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.text import Tokenizer
# noinspection PyUnresolvedReferences
import pickle
import pandas as pd
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Sequential
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.sequence import pad_sequences
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.text import Tokenizer
from config import *

def get_training_data(file_name):
    data = pd.read_csv(file_name, encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    return [data['label'].values], data['message'].values

def get_training_data2(file_path):
    data = pd.read_csv(file_path, sep=',', header=0, names=['text_type', 'text'])
    data.columns = ['text_type', 'text']
    data['text_type'] = data['text_type'].map({'ham': 0, 'spam': 1})
    return [data['text_type'].values], data['text'].values

def get_training_data_json(file_name):
    data = pd.read_json(file_name)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return [data['label'].values], data['message'].values

# labels, texts = get_learn_data(csv_file)
# labels, texts = get_learn_data2(csv_file_hugging)
labels, texts = get_training_data_json(json_file)

vocab_size = 5000
max_length = 50
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(texts)

X_train = tokenizer.texts_to_sequences(texts)
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
y_train = labels

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=16, callbacks=[early_stop])


# model.save('dl_spam_model.keras')
# with open('../model_uci/dl_tokenizer.pkl', 'wb') as f:
#     pickle.dump(tokenizer, f)

# model.save('../model_hugging/dl_spam_model2.keras')
# with open('../model_hugging/dl_tokenizer2.pkl', 'wb') as f:
#     pickle.dump(tokenizer, f)

model.save('../model_small_json/dl_spam_model3.keras')
with open('../model_small_json/dl_tokenizer3.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Deep learning model and tokenizer saved.")