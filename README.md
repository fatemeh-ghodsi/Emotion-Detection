#Emotion Detection using Bidirectional LSTM
This project is a Deep Learning-based NLP application that classifies text into emotional categories (such as joy, fear, anger, etc.) using the ISEAR (International Survey on Emotion Antecedents and Reactions) dataset. It leverages spaCy for advanced preprocessing and a Stacked Bidirectional LSTM for high-accuracy sequence modeling.

🚀 Overview
The core of this project is a neural network designed to understand the context and sentiment of human language. By processing text in both forward and backward directions, the model captures nuances that simpler models might miss.

🛠️ Tech Stack
Language: Python 3.x

NLP: spaCy (en_core_web_sm)

Deep Learning: TensorFlow 2.x / Keras

Data Processing: Pandas, NumPy, Scikit-learn

🏗️ Model Architecture
The model is built with the following layers to ensure robust feature extraction and prevent overfitting:

Embedding Layer: Maps words to dense vectors of 50 dimensions.

SpatialDropout1D (0.3): Prevents overfitting by dropping entire 1D feature maps.

Stacked Bi-LSTM: * Layer 1: 128 units (returns sequences).

Layer 2: 64 units.

Batch Normalization: Accelerates training and provides stability.

Dropout (0.5): Further regularizes the network.

Dense Output: Uses Softmax activation with L2 Regularization for multi-class classification.

📋 Prerequisites
Before running the code, install the required dependencies:

Bash

pip install numpy pandas spacy tensorflow scikit-learn
python -m spacy download en_core_web_sm
📂 Dataset Structure
The script is designed for the isear.csv dataset. It expects:

Column 0: Emotion Labels.

Column 1: Textual data.

The script automatically filters out rows with "No response" to ensure high-quality training data.

⚙️ Usage
Prepare your data: Ensure your isear.csv is in the correct directory (update the data_path in train.py if necessary).

Train the model:

Bash

python train.py
Monitoring: The script includes an EarlyStopping callback that monitors val_loss and stops training once the model stops improving, restoring the best weights automatically.

📊 Performance Features
Custom Tokenization: Combines spaCy's linguistic precision with Keras's sequence processing.

Class Encoding: Uses One-Hot Encoding for multi-class label representation.

Optimization: Uses the Adam optimizer with a learning rate of 0.001.
