import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import os
import shutil
import glob

# ==========================================
# CONFIGURATION
# ==========================================
PYTORCH_MODEL_PATH = "./final_scam_model"
TF_SAVED_MODEL_DIR = "./tf_intermediate_model"
TFLITE_OUTPUT_FILE = "scam_detection_quantized.tflite"

# ==========================================
# 1. LOAD PYTORCH MODEL & CONVERT TO TF
# ==========================================
print(f"[1/4] Loading PyTorch model from {PYTORCH_MODEL_PATH}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_PATH)
    # Load and convert PyTorch weights to TensorFlow on-the-fly
    model = TFAutoModelForSequenceClassification.from_pretrained(
        PYTORCH_MODEL_PATH, 
        from_pt=True
    )
except OSError:
    print(f"ERROR: Could not find folder '{PYTORCH_MODEL_PATH}'.")
    exit()

print("[2/4] Saving intermediate TensorFlow model...")
# This creates a folder structure like: ./tf_intermediate_model/saved_model/1/saved_model.pb
model.save_pretrained(TF_SAVED_MODEL_DIR, saved_model=True)

# ==========================================
# 2. FIND THE CORRECT SAVED_MODEL PATH
# ==========================================
# Hugging Face hides the .pb file inside subfolders. We find it automatically.
print("      Locating saved_model.pb...")
pb_files = glob.glob(f"{TF_SAVED_MODEL_DIR}/**/saved_model.pb", recursive=True)

if not pb_files:
    print("ERROR: saved_model.pb was not generated. Quantization cannot proceed.")
    exit()

# The converter needs the *directory* containing saved_model.pb, not the file itself
real_saved_model_dir = os.path.dirname(pb_files[0])
print(f"      Found model at: {real_saved_model_dir}")

# ==========================================
# 3. QUANTIZATION SETUP (INT8)
# ==========================================
print(f"[3/4] Compressing model from {real_saved_model_dir}...")

# Point converter to the detected folder
converter = tf.lite.TFLiteConverter.from_saved_model(real_saved_model_dir)

# ENABLE DYNAMIC RANGE QUANTIZATION
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define dummy input generator for calibration
def representative_dataset_gen():
    for _ in range(20):
        # Shape: (1, 128) -> Standard for mobile inference
        # Use int64/int32 based on what the tokenizer usually outputs
        dummy_input = tf.random.uniform((1, 128), minval=0, maxval=tokenizer.vocab_size, dtype=tf.int64)
        yield [dummy_input, dummy_input] 

# ==========================================
# 4. CONVERT & SAVE
# ==========================================
try:
    tflite_model = converter.convert()
except Exception as e:
    print("\nError during TFLite conversion. Trying to remove input signature restrictions...")
    # Fallback: Sometimes specific input shapes cause issues, retrying without specific signature can help
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

with open(TFLITE_OUTPUT_FILE, "wb") as f:
    f.write(tflite_model)

# ==========================================
# 5. REPORT & CLEANUP
# ==========================================
print("\n" + "="*50)
print("      QUANTIZATION SUCCESSFUL")
print("="*50)

# Calculate Sizes
original_weights = os.path.join(PYTORCH_MODEL_PATH, "model.safetensors")
if os.path.exists(original_weights):
    orig_size = os.path.getsize(original_weights) / (1024 * 1024)
else:
    orig_size = os.path.getsize(os.path.join(PYTORCH_MODEL_PATH, "pytorch_model.bin")) / (1024 * 1024)

quant_size = os.path.getsize(TFLITE_OUTPUT_FILE) / (1024 * 1024)
reduction = (1 - (quant_size / orig_size)) * 100

print(f"Original Model:   {orig_size:.2f} MB")
print(f"Quantized Model:  {quant_size:.2f} MB")
print(f"Reduction:        {reduction:.2f}% (Compressed)")
print("-" * 50)
print(f"OUTPUT SAVED:     {os.path.abspath(TFLITE_OUTPUT_FILE)}")
print("="*50)

# Cleanup
if os.path.exists(TF_SAVED_MODEL_DIR):
    shutil.rmtree(TF_SAVED_MODEL_DIR)