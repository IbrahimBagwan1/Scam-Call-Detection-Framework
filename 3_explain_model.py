import shap
import transformers
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "./final_scam_model"

# Strong scam text to ensure high confidence
scam_text = [
    "Hello sir, this is the bank manager calling. We have detected suspicious activity on your credit card. To stop this transaction and unblock your funds, you must verify your identity. Please provide the One Time Password OTP sent to your mobile number immediately or your account will be frozen permanently."
]

# ==========================================
# 1. LOAD MODEL
# ==========================================
print("Loading model...")
try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
except:
    print("Error: Model not found. Make sure you are in the right folder.")
    exit()

device = 0 if torch.cuda.is_available() else -1
if device == 0:
    model.to("cuda")

# ==========================================
# 2. RUN SHAP ANALYSIS
# ==========================================
print("Running SHAP analysis...")
pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)
explainer = shap.Explainer(pred)
shap_values = explainer(scam_text)

# ==========================================
# 3. EXTRACT DATA MANUALLY (Crash-Proof)
# ==========================================
# We grab the scores for Class 1 (Scam)
# shap_values[0] = the first sentence
# .values = the numerical scores
# .data = the actual words (tokens)
scores = shap_values[0, :, 1].values
words = shap_values[0, :, 1].data

# Create a DataFrame to sort easily
df = pd.DataFrame({
    'word': words,
    'score': scores
})

# Filter out empty spaces or tiny tokens to clean up the graph
df = df[df['word'].str.strip() != ''] 

# Sort by impact (High scores = Scam indicators)
df_sorted = df.sort_values(by='score', ascending=True).tail(10)  # Get Top 10

# ==========================================
# 4. DRAW THE CHART (Matplotlib)
# ==========================================
print("Generating clean IEEE Bar Chart...")

plt.figure(figsize=(10, 6))

# Create Horizontal Bar Chart
bars = plt.barh(df_sorted['word'], df_sorted['score'], color='#ff4d4d') # Red color for danger

# Formatting for IEEE Paper
plt.xlabel('SHAP Value (Impact on Scam Prediction)', fontsize=12, fontweight='bold')
plt.title('Top 10 Words Triggering "Scam" Detection', fontsize=14, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels on the bars
for index, value in enumerate(df_sorted['score']):
    plt.text(value, index, f' +{value:.2f}', va='center', fontweight='bold')

plt.tight_layout()

# Save
output_file = "Fig5_SHAP_BarChart.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print(f"\nSUCCESS! Chart saved to: {os.path.abspath(output_file)}")
print("Open this image and put it in your paper as Figure 5.")