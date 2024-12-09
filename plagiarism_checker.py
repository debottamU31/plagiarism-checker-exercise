# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( classification_report, accuracy_score)
from transformers import BertTokenizer, TFBertModel
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("bigdataset.csv")

# Get the text data and labels
texts = data["text"].tolist()
labels = data["label"].tolist()

# Split data into training (80%) and test (20%) sets, maintaining the label balance
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,  # 20% for testing
    random_state=42,
    stratify=labels,  # Maintain balance in labels
)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the texts
input_ids = []
attention_masks = []

for text in texts:
    encoded_data = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
        return_tensors="tf",
    )
    input_ids.append(encoded_data["input_ids"])
    attention_masks.append(encoded_data["attention_mask"])

# Convert lists to tensors
input_ids = tf.concat(input_ids, axis=0).numpy()
attention_masks = tf.concat(attention_masks, axis=0).numpy()
labels = tf.convert_to_tensor(labels).numpy()

# Split data into training and test sets
X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = (
    train_test_split(input_ids, attention_masks, labels, test_size=0.2, random_state=42)
)

# Load the BERT model (with trainable layers)
model_name = "bert-base-uncased"
bert_model = TFBertModel.from_pretrained(model_name)

# Define the input layers for the tokenized data
input_ids_layer = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
attention_mask_layer = tf.keras.layers.Input(
    shape=(512,), dtype=tf.int32, name="attention_mask"
)

# Fine-tune the BERT model by keeping its layers trainable
bert_outputs = bert_model(input_ids_layer, attention_mask=attention_mask_layer)

# Use the pooled output for classification
pooled_output = bert_outputs.pooler_output  # This is (batch_size, hidden_size)

# Add a dropout layer for regularization
dropout_layer = tf.keras.layers.Dropout(0.3)(pooled_output)

# Output layer for binary classification
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(dropout_layer)

# Create the model
model = tf.keras.Model(
    inputs=[input_ids_layer, attention_mask_layer], outputs=output_layer
)

# Compile the model with optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Model summary
model.summary()
# Define callbacks (Early stopping)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
]

# Train the model with training data
history = model.fit(
    x={"input_ids": X_train_ids, "attention_mask": X_train_masks},
    y=y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=16,
    callbacks=callbacks,
)
# Evaluate the model
y_pred_probs = model.predict({"input_ids": X_test_ids, "attention_mask": X_test_masks})
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Print classification report
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
# Save model as HDF5 file
model.save("my_model_v3_new.h5")

# Load the trained model (make sure the model is loaded or you can load from the saved file)
# model = tf.keras.models.load_model('my_model')

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def predict_text(model, text, max_length=128):
    # Preprocess the input text (tokenize and pad)
    encoded_data = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="tf",
    )

    input_ids = encoded_data["input_ids"]
    attention_mask = encoded_data["attention_mask"]

    # Make prediction
    prediction = model.predict(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )

    print(f"Prediction is: {prediction}")

    # Convert prediction to label (Assuming binary classification: 0 for human, 1 for AI)
    label = "AI-generated" if prediction > 0.4 else "Human-written"
    return label


# Example of manually inputting a text
input_text = input("Enter the text to classify: ")
result = predict_text(model, input_text)

print(f"The text is classified as: {result}")
