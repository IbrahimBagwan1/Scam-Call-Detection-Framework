import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
PRETRAINED_PATH = "./sms_pretrained_model"
FILE_NAME = "dataset\call_dataset.csv"

# ==========================================
# STEP 1: ROBUST DATA LOADING (Fixes KeyError)
# ==========================================
print(f"Loading Call Transcripts from {FILE_NAME}...")

if not os.path.exists(FILE_NAME):
    print(f"CRITICAL ERROR: {FILE_NAME} not found! Please check the file name.")
    exit()

try:
    # Try latin-1 first (handles special chars best)
    df = pd.read_csv(FILE_NAME, encoding='latin-1')
except:
    # Fallback to standard utf-8
    df = pd.read_csv(FILE_NAME)

print(f"Original Columns: {df.columns.tolist()}")

# 1. Normalize Column Names (Lowercase + Strip spaces)
# This fixes 'Transcript' vs 'transcript' vs ' transcript '
df.columns = [c.strip().lower() for c in df.columns]

# 2. Rename to 'text' (The code needs this)
if 'transcript' in df.columns:
    df = df.rename(columns={'transcript': 'text'})
elif 'content' in df.columns:
    df = df.rename(columns={'content': 'text'})
elif 'sentence' in df.columns:
    df = df.rename(columns={'sentence': 'text'})

# 3. Rename to 'label' (The code needs this)
if 'labels' in df.columns:
    df = df.rename(columns={'labels': 'label'})
elif 'class' in df.columns:
    df = df.rename(columns={'class': 'label'})
elif 'target' in df.columns:
    df = df.rename(columns={'target': 'label'})

# 4. Verify we succeeded
if 'text' not in df.columns or 'label' not in df.columns:
    print("\nCRITICAL ERROR: Could not find 'Transcript' or 'Labels' columns.")
    print(f"Detected columns after cleaning: {df.columns.tolist()}")
    exit()

# 5. Clean Data (Now safe to do)
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)

# Split 80/20
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ==========================================
# STEP 2: LOAD THE SMS-TRAINED MODEL
# ==========================================
print("Loading your custom SMS-trained model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_PATH, num_labels=2)
except OSError:
    print("\nERROR: 'sms_pretrained_model' folder is missing.")
    print("You must run Step 1 successfully before this step.")
    exit()

def tokenize_function(examples):
    # Call transcripts are long, so we use max_length=512
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

print("Tokenizing Call Transcripts...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ==========================================
# STEP 3: METRICS FUNCTION
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# ==========================================
# STEP 4: FINE-TUNING SETUP (RTX 4060 Optimized)
# ==========================================
training_args = TrainingArguments(
    output_dir="./results_calls",
    num_train_epochs=5,              # 5 Epochs for high accuracy
    per_device_train_batch_size=8,   # Batch size 8 fits your 8GB VRAM
    learning_rate=1e-5,              # Low rate to keep SMS knowledge
    eval_strategy="epoch",           # Updated keyword (Fixed error)
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                       # Fast training on RTX 4060
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# ==========================================
# STEP 5: TRAIN & SAVE
# ==========================================
print("STARTING CALL FINE-TUNING...")
trainer.train()

print("Saving Final Research Model...")
model.save_pretrained("./final_scam_model")
tokenizer.save_pretrained("./final_scam_model")

# Print Final Numbers for your Paper
results = trainer.evaluate()
print("\n" + "="*40)
print("FINAL RESULTS FOR IEEE PAPER")
print("="*40)
print(f"ACCURACY : {results['eval_accuracy']:.4f}")
print(f"F1 SCORE : {results['eval_f1']:.4f}")
print("="*40)