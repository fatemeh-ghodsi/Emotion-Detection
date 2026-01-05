# train.py

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


def load_and_clean_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    print('Before removing extra rows:', df.shape)
    # Remove rows with "No response" in the second column (case insensitive)
    df.drop(df[df.iloc[:, 1].astype(str).str.lower().str.contains(r'no\s*response', na=False)].index, inplace=True)
    print('After removing "No response" rows:', df.shape)
    df = df.reset_index(drop=True)
    return df


def tokenize_texts(texts, nlp):
    # Tokenize texts with spaCy, lowercase, remove punctuation and spaces
    docs = nlp.pipe(texts, batch_size=50, disable=['parser', 'ner'])
    tokenized_texts = [
        [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        for doc in docs
    ]
    # Join tokens back to strings for Keras Tokenizer
    return [' '.join(tokens) for tokens in tokenized_texts]


def main():
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load spaCy model with disabled unnecessary components for speed
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Update this path to your dataset location
    data_path = r"D:\jangoProject\nlp\EmotionDetection\isear.csv"

    # Load and clean data
    df = load_and_clean_data(data_path)

    # Extract texts and labels
    texts = df.iloc[:, 1].tolist()
    labels = df.iloc[:, 0].values.reshape(-1, 1)

    # Tokenize and preprocess texts
    cleaned_texts = tokenize_texts(texts, nlp)

    # Tokenize texts with Keras Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cleaned_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)

    max_length = 100  # max sequence length for padding
    X = pad_sequences(sequences, maxlen=max_length, padding='post')

    # One-hot encode labels
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    y = encoder.fit_transform(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Model parameters
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    num_classes = y.shape[1]

    # Build model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    model.summary()

    # Setup early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Fit model with validation on test data (consider splitting train for validation if desired)
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate model on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    main()
