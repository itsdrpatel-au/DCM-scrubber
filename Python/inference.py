import os
import json
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pydicom
import pytesseract
import tkinter as tk
from tkinter import filedialog

############################################
# Utility Functions
############################################

def convert_dicom_to_pil(dicom_path):
    """
    Reads a DICOM file and converts it to a PIL Image.
    Normalizes the pixel data to an 8-bit RGB image.
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        img_array = ds.pixel_array.astype("float")
        img_array = 255 * (img_array - img_array.min()) / (img_array.ptp() + 1e-8)
        img_array = img_array.astype("uint8")
        pil_img = Image.fromarray(img_array)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return pil_img
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")
        return None

def load_annotation_text(dicom_path):
    """
    For a given DICOM file, loads the corresponding annotation text file
    (with filename <basename>_annotation.txt) from the same folder.
    If not found, falls back to OCR extraction.
    """
    folder, filename = os.path.split(dicom_path)
    base, _ = os.path.splitext(filename)
    ann_filename = f"{base}_annotation.txt"
    ann_path = os.path.join(folder, ann_filename)
    if os.path.exists(ann_path):
        with open(ann_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        pil_img = convert_dicom_to_pil(dicom_path)
        if pil_img is None:
            return ""
        ocr_text = pytesseract.image_to_string(pil_img)
        return ocr_text.strip()

############################################
# Dummy Tokenizer
############################################

class DummyTokenizer:
    """
    A basic whitespace tokenizer that builds its vocabulary on the fly.
    In production you should use the same tokenizer/vocabulary used during training.
    """
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4

    def build_vocab(self, texts):
        for text in texts:
            for word in text.split():
                word = word.lower()
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        self.vocab_size = len(self.word2idx)

    def tokenize(self, text, max_length=300):
        tokens = [self.word2idx.get(word.lower(), self.word2idx["<UNK>"]) for word in text.split()]
        tokens = [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]
        if len(tokens) < max_length:
            tokens = tokens + [self.word2idx["<PAD>"]] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids):
        words = []
        for idx in token_ids:
            word = self.idx2word.get(int(idx), "<UNK>")
            if word == "<EOS>":
                break
            if word not in ["<SOS>", "<PAD>"]:
                words.append(word)
        return " ".join(words)

############################################
# Model Definitions
############################################

# 1. Image Encoder using DenseNet121
class ImageEncoder(nn.Module):
    def __init__(self, use_radimagenet=False, rad_weights_path=None):
        super(ImageEncoder, self).__init__()
        if use_radimagenet and rad_weights_path and os.path.exists(rad_weights_path):
            print("Loading RadImageNet DenseNet121 weights...")
            model = models.densenet121(pretrained=False)
            state_dict = torch.load(rad_weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        else:
            print("Loading standard DenseNet121 weights...")
            model = models.densenet121(pretrained=True)
        self.features = model.features  # (batch, 1024, H, W)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        feat = self.features(x)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)  # (batch, 1024)
        return feat

# 2. Annotation Encoder for per-image OCR text
class AnnotationEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=1):
        super(AnnotationEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, token_seq):
        embedded = self.embedding(token_seq)  # (batch, seq_length, embed_size)
        _, (h_n, _) = self.lstm(embedded)
        return h_n[-1]  # (batch, hidden_size)

# 3. Report Decoder to generate final report text
class ReportDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=256, num_layers=1):
        super(ReportDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embedded = self.embedding(captions)  # (batch, seq_length, embed_size)
        h0 = features.unsqueeze(0)  # (1, batch, hidden_size)
        c0 = torch.zeros_like(h0)   # (1, batch, hidden_size)
        outputs, _ = self.lstm(embedded, (h0, c0))
        outputs = self.fc(outputs)
        return outputs

    def generate(self, features, max_length=100):
        batch_size = features.size(0)
        h0 = features.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        input_token = torch.full((batch_size, 1), tokenizer.word2idx["<SOS>"], dtype=torch.long, device=features.device)
        generated_tokens = []
        hidden = (h0, c0)
        for _ in range(max_length):
            embedded = self.embedding(input_token)
            output, hidden = self.lstm(embedded, hidden)
            output = self.fc(output)  # (batch, 1, vocab_size)
            token = output.argmax(dim=-1)  # (batch, 1)
            generated_tokens.append(token)
            input_token = token
            if (token == tokenizer.word2idx["<EOS>"]).all():
                break
        generated_tokens = torch.cat(generated_tokens, dim=1)
        return generated_tokens

# 4. Multi-Modal Model: Fuses image and annotation features to generate final report.
class MultiModalModel(nn.Module):
    def __init__(self, vocab_size, use_radimagenet=False, rad_weights_path=None):
        super(MultiModalModel, self).__init__()
        self.image_encoder = ImageEncoder(use_radimagenet, rad_weights_path)
        self.annotation_encoder = AnnotationEncoder(vocab_size)
        # For each image-annotation pair, we concatenate image features (1024-d) with annotation features (256-d)
        # to get a 1280-d vector. We average these vectors over all images in the study.
        self.fusion_fc = nn.Linear(1280, 256)
        self.decoder = ReportDecoder(vocab_size)
    
    def forward(self, images, annotations_tokens, report_tokens):
        # images: list of image tensors (one per image in a study)
        # annotations_tokens: list of token tensors corresponding to each image's OCR text
        pair_features = []
        for img, ann_tokens in zip(images, annotations_tokens):
            img_feat = self.image_encoder(img.unsqueeze(0))  # (1, 1024)
            ann_feat = self.annotation_encoder(ann_tokens.unsqueeze(0))  # (1, 256)
            pair_feat = torch.cat([img_feat, ann_feat], dim=1)  # (1, 1280)
            pair_features.append(pair_feat)
        # Average across all image-annotation pairs.
        study_feature = torch.mean(torch.cat(pair_features, dim=0), dim=0, keepdim=True)  # (1, 1280)
        fused_features = torch.tanh(self.fusion_fc(study_feature))  # (1, 256)
        outputs = self.decoder(fused_features, report_tokens)
        return outputs

    def generate(self, images, annotations_tokens, max_length=100):
        pair_features = []
        for img, ann_tokens in zip(images, annotations_tokens):
            img_feat = self.image_encoder(img.unsqueeze(0))
            ann_feat = self.annotation_encoder(ann_tokens.unsqueeze(0))
            pair_feat = torch.cat([img_feat, ann_feat], dim=1)
            pair_features.append(pair_feat)
        study_feature = torch.mean(torch.cat(pair_features, dim=0), dim=0, keepdim=True)
        fused_features = torch.tanh(self.fusion_fc(study_feature))
        generated_tokens = self.decoder.generate(fused_features, max_length)
        return generated_tokens

############################################
# Inference: Generate Report.json from a Study Folder
############################################

def generate_report_for_study(study_folder, model, tokenizer, device, image_transform, max_text_length=300, max_report_length=100):
    """
    Given a study folder containing DICOM images and per-image annotation files,
    extracts images and annotations, passes them through the multi-modal model,
    and generates a final report which is saved as Report.json in the study folder.
    """
    images = []
    annotations_tokens = []
    
    for fname in os.listdir(study_folder):
        if fname.lower().endswith('.dcm'):
            dicom_path = os.path.join(study_folder, fname)
            # Load image tensor.
            pil_img = convert_dicom_to_pil(dicom_path)
            if pil_img is None:
                continue
            img_tensor = image_transform(pil_img)
            images.append(img_tensor)
            # Load corresponding annotation text.
            base, _ = os.path.splitext(fname)
            ann_fname = f"{base}_annotation.txt"
            ann_path = os.path.join(study_folder, ann_fname)
            if os.path.exists(ann_path):
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann_text = f.read().strip()
            else:
                ann_text = pytesseract.image_to_string(pil_img) if pil_img is not None else ""
            tokens = tokenizer.tokenize(ann_text, max_length=max_text_length)
            annotations_tokens.append(tokens)
    
    if len(images) == 0:
        print("No images found in the study folder.")
        return
    
    model.eval()
    with torch.no_grad():
        # Our model expects a list of image tensors and a list of annotation token tensors.
        generated_tokens = model.generate(images, annotations_tokens, max_length=max_report_length)
    report_str = tokenizer.decode(generated_tokens[0])
    print("Generated Report:")
    print(report_str)
    
    # Attempt to parse the report as JSON.
    try:
        report_json = json.loads(report_str)
    except Exception as e:
        print("Warning: Generated report is not valid JSON. Saving as plain text.")
        report_json = {"report": report_str}
    
    output_path = os.path.join(study_folder, "Report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    print(f"Generated report saved to {output_path}")

############################################
# Main Inference Routine with Dialog Box
############################################

if __name__ == "__main__":
    # Set device (using CPU for now)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create a dialog box to select the study folder.
    root = tk.Tk()
    root.withdraw()
    study_folder = filedialog.askdirectory(title="Select Study Folder Containing DICOM Images")
    if not study_folder:
        print("No folder selected. Exiting.")
        exit(1)
    print(f"Selected study folder: {study_folder}")
    
    # Define image transforms.
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize the dummy tokenizer.
    tokenizer = DummyTokenizer()
    # Load the checkpoint to update the vocabulary.
    checkpoint_path = os.path.join("checkpoints", "checkpoint_epoch_10.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Update tokenizer with vocabulary from the checkpoint.
        tokenizer.word2idx = checkpoint["tokenizer_word2idx"]
        tokenizer.idx2word = checkpoint["tokenizer_idx2word"]
        tokenizer.vocab_size = len(tokenizer.word2idx)
    else:
        print("Checkpoint not found, using current tokenizer (vocab size may be small).")
    
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)
    
    # Instantiate the multi-modal model.
    rad_weights_path = "weights/radimagenet_densenet121.pth"  # Update if using RadImageNet weights.
    model = MultiModalModel(tokenizer.vocab_size, use_radimagenet=False, rad_weights_path=rad_weights_path).to(device)
    
    # Optionally, load the model checkpoint if available.
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Model checkpoint not found, using model with random weights (for testing).")
    
    # Generate the final report for the selected study.
    generate_report_for_study(study_folder, model, tokenizer, device, image_transform, max_text_length=300, max_report_length=100)
