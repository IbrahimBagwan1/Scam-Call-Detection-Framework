import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "distilroberta-base"
FILE_NAME = "dataset\sms_dataset.csv"

# ==========================================
# STEP 1: LOAD DATA (ROBUST VERSION)
# ==========================================
print(f"Loading data from {FILE_NAME}...")

try:
    # Use latin-1 to handle special characters in SMS
    df = pd.read_csv(FILE_NAME, encoding='latin-1')
except UnicodeDecodeError:
    df = pd.read_csv(FILE_NAME)

# Fix Column Names (Handle 'labels' vs 'label')
if 'labels' in df.columns:
    df = df.rename(columns={"transcript": "text", "labels": "label"})
elif 'transcript' in df.columns:
    df = df.rename(columns={"transcript": "text", "label": "label"})

# Clean Data (Drop empty rows and ensure labels are numbers)
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)

# Split 80/20
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ==========================================
# STEP 2: PREPARE MODEL
# ==========================================
print("Downloading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizing data...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ==========================================
# STEP 3: TRAIN
# ==========================================
training_args = TrainingArguments(
    output_dir="./results_sms",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch",           # <--- UPDATED: This is the correct new keyword
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

print("STARTING TRAINING...")
trainer.train()

# ==========================================
# STEP 4: SAVE
# ==========================================
print("Saving model to 'sms_pretrained_model'...")
model.save_pretrained("./sms_pretrained_model")
tokenizer.save_pretrained("./sms_pretrained_model")
print("SUCCESS! Step 1 Complete.")