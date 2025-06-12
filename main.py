import torch
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Step 1: Load GoEmotions dataset and use subset
def load_goemotions_subset(n=1000):
    print("Loading GoEmotions subset...")
    dataset = load_dataset("go_emotions", "raw")["train"]
    df = pd.DataFrame(dataset)
    df = df.head(n)  # use only first 1000 rows
    return df

# Step 2: Preprocess multi-labels
def preprocess(df):
    print("Preparing multi-labels...")
    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    labels = df[emotion_columns].values
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    return df["text"].tolist(), y, mlb

# Step 3: Tokenize texts
def tokenize_texts(texts, tokenizer, max_length=128):
    print("Tokenizing text...")
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Step 4: Custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Step 5: Load DistilBERT model
def get_model(num_labels):
    print("Loading DistilBERT model...")
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

# Step 6: Real-world inference
def predict_emotions(model, tokenizer, mlb, text):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        output = model(**tokens)
        probs = torch.sigmoid(output.logits)
        pred_labels = (probs > 0.5).int().squeeze().tolist()
    return [mlb.classes_[i] for i, val in enumerate(pred_labels) if val]

# Main execution
if __name__ == "__main__":
    df = load_goemotions_subset(n=1000)
    texts, labels, mlb = preprocess(df)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized = tokenize_texts(texts, tokenizer)
    dataset = EmotionDataset(tokenized, labels)

    model = get_model(num_labels=len(mlb.classes_))

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,  # just 1 epoch for fast training
        per_device_train_batch_size=16,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="no",
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    print("Training...")
    trainer.train()

    print("Evaluating...")
    outputs = trainer.predict(dataset)
    preds = (torch.sigmoid(torch.tensor(outputs.predictions)) > 0.5).int().numpy()

    print("Hamming Loss:", hamming_loss(labels, preds))
    print("F1 Score (micro):", f1_score(labels, preds, average='micro'))

    # Real-world example
    example_text = "I'm so excited and happy today!"
    print("\nInput:", example_text)
    print("Predicted Emotions:", predict_emotions(model, tokenizer, mlb, example_text))
