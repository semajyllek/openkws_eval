import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from pathlib import Path
import os
import glob
from tqdm import tqdm
import numpy as np

# Import constants from our new config file
from config.eval_params import (
    MODEL_NAME, FINE_TUNED_MODEL_PATH, DEVICE, CUSTOM_WAKE_WORD_DIR,
    GSC_DATASET_PATH, GSC_WORDS, NON_WAKE_WORD_LABEL, WAKE_WORD_LABEL, BATCH_SIZE
)

# --- Custom Dataset for Evaluation ---
class EvalDataset(Dataset):
    """A custom dataset to handle audio data with binary labels."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        waveform = item['audio']
        return {
            "audio": waveform,
            "labels": torch.tensor(item['label'], dtype=torch.long)
        }

# --- Data Preparation Functions ---
def prepare_gsc_data():
    """Programmatically loads and filters the Google Speech Commands dataset."""
    print("Preparing Google Speech Commands data...")
    dataset = torchaudio.datasets.SPEECHCOMMANDS(root=GSC_DATASET_PATH, download=True)
    
    filtered_data = []
    
    for _, _, label, _, _ in tqdm(dataset, desc="Filtering GSC data"):
        if label in GSC_WORDS:
            filtered_data.append({
                "audio": torchaudio.load(dataset._walker[dataset._hash_func(label, dataset.sub_dir_name)], normalize=True),
                "label": NON_WAKE_WORD_LABEL
            })
    
    print(f"Loaded {len(filtered_data)} samples from GSC.")
    return filtered_data

def prepare_custom_data():
    """Loads custom wake word audio files from the local directory."""
    print("Preparing custom wake word data...")
    audio_paths = sorted(glob.glob(os.path.join(CUSTOM_WAKE_WORD_DIR, "*.wav")))
    
    data = []
    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        data.append({"audio": waveform.squeeze(), "label": WAKE_WORD_LABEL})

    print(f"Loaded {len(audio_paths)} custom wake word samples.")
    return data

def prepare_evaluation_dataloaders():
    """Creates the two evaluation datasets and dataloaders."""
    
    # Get all the raw data
    ood_data = prepare_custom_data()
    gsc_data = prepare_gsc_data()
    
    # 1. DataLoader for the OOD (Out-of-Dataset) Test
    ood_test_data = ood_data + gsc_data
    ood_dataset = EvalDataset(ood_test_data)
    ood_dataloader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. DataLoader for the ID (In-Dataset) Test
    num_positive_id = 1000
    id_positive_data = gsc_data[:num_positive_id]
    id_negative_data = gsc_data[num_positive_id:]
    
    for item in id_positive_data:
        item['label'] = WAKE_WORD_LABEL
    
    id_test_data = id_positive_data + id_negative_data
    id_dataset = EvalDataset(id_test_data)
    id_dataloader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return ood_dataloader, id_dataloader

def evaluate_model(model, dataloader, processor, name):
    """Evaluates a model's binary classification accuracy."""
    print(f"\n--- Evaluating {name} ---")
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = processor(
                raw_speech=[x.numpy() for x in batch['audio']],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(inputs.input_values)
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print(f"{name} Accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_baseline():
    """Performs the pre-trained model baseline evaluation."""
    ood_dataloader, id_dataloader = prepare_evaluation_dataloaders()
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    print("\n##### Stage 1: Pre-trained Model Baseline #####")
    model_pretrained = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    
    id_baseline_accuracy = evaluate_model(model_pretrained, id_dataloader, processor, "Pre-trained on In-Dataset (GSC) Words")
    ood_baseline_accuracy = evaluate_model(model_pretrained, ood_dataloader, processor, "Pre-trained on Out-of-Dataset (Custom) Words")
    
    return id_baseline_accuracy, ood_baseline_accuracy

def evaluate_finetuned():
    """Performs the fine-tuned model evaluation, if the model exists."""
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        print(f"\nSkipping fine-tuned model evaluation: '{FINE_TUNED_MODEL_PATH}' not found.")
        return None, None
        
    ood_dataloader, id_dataloader = prepare_evaluation_dataloaders()
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    
    print("\n##### Stage 2: Fine-tuned Model Performance #####")
    model_finetuned = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model_finetuned.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
    model_finetuned.to(DEVICE)

    id_finetuned_accuracy = evaluate_model(model_finetuned, id_dataloader, processor, "Fine-tuned on In-Dataset (GSC) Words")
    ood_finetuned_accuracy = evaluate_model(model_finetuned, ood_dataloader, processor, "Fine-tuned on Out-of-Dataset (Custom) Words")
    
    return id_finetuned_accuracy, ood_finetuned_accuracy

if __name__ == "__main__":
    evaluate_baseline()
