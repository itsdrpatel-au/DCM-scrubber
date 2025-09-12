import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pydicom
import pytesseract
from transformers import GPT2Tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################
# Utility Functions
############################################

def convert_dicom_to_pil(dicom_path):
    """
    Reads a DICOM file and converts it into a PIL Image.
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
        logger.error(f"Error converting {dicom_path}: {e}")
        return None

def ocr_extract(dicom_path):
    """
    Extracts OCR text from a DICOM file.
    """
    image = convert_dicom_to_pil(dicom_path)
    if image is None:
        return ""
    return pytesseract.image_to_string(image).strip()

def flatten_report_json(report_data):
    """
    Flattens a nested JSON report into a multi-line string with "key: value" lines.
    """
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, ", ".join(str(item) for item in v)))
            else:
                items.append((new_key, str(v)))
        return dict(items)
    flat = flatten_dict(report_data)
    lines = [f"{k}: {v}" for k, v in sorted(flat.items())]
    return "\n".join(lines)

def gpt2_tokenize(text, tokenizer, max_length=300):
    """
    Tokenizes text using the provided GPT2Tokenizer with padding and truncation.
    Returns a tensor of token IDs.
    """
    encoded = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, 
                                 truncation=True, padding='max_length')
    return torch.tensor(encoded, dtype=torch.long)

############################################
# Dataset: MultiModalStudyDataset
############################################

class MultiModalStudyDataset(Dataset):
    """
    Each sample corresponds to one study folder in pilot_dataset.
    For each study, returns:
      - images: a list of image tensors (one per DICOM file)
      - annotations_tokens: a list of token tensors (one per image, from corresponding annotation files)
      - report_tokens: a token tensor for the final report (flattened from Report.json)
    """
    def __init__(self, root_dir, image_transform, tokenizer, max_text_length=300):
        self.root_dir = root_dir
        self.study_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))]
        self.transform = image_transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.study_folders)

    def __getitem__(self, idx):
        study_folder = self.study_folders[idx]
        
        images = []
        annotations_texts = []
        for fname in os.listdir(study_folder):
            if fname.lower().endswith('.dcm'):
                dicom_path = os.path.join(study_folder, fname)
                pil_img = convert_dicom_to_pil(dicom_path)
                if pil_img is not None:
                    img_tensor = self.transform(pil_img)
                    images.append(img_tensor)
                base, _ = os.path.splitext(fname)
                ann_fname = f"{base}_annotation.txt"
                ann_path = os.path.join(study_folder, ann_fname)
                if os.path.exists(ann_path):
                    with open(ann_path, "r", encoding="utf-8") as f:
                        ann_text = f.read().strip()
                else:
                    ann_text = ocr_extract(dicom_path)
                annotations_texts.append(ann_text)
        
        if len(images) == 0:
            images.append(torch.zeros(3, 224, 224))
            annotations_texts.append("")
        
        annotations_tokens = [gpt2_tokenize(text, self.tokenizer, self.max_text_length)
                              for text in annotations_texts]
        
        report_path = os.path.join(study_folder, "Report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            report_text = flatten_report_json(report_data)
        else:
            report_text = ""
        report_tokens = gpt2_tokenize(report_text, self.tokenizer, self.max_text_length)
        
        return {
            "images": images,
            "annotations_tokens": annotations_tokens,
            "report_tokens": report_tokens
        }

def collate_fn(samples):
    """
    Custom collate function for a batch of studies.
    Returns:
      - images: list (length=batch) of lists of image tensors.
      - annotations_tokens: list (length=batch) of lists of token tensors.
      - report_tokens: tensor (batch, seq_length).
    """
    images = [s["images"] for s in samples]
    annotations_tokens = [s["annotations_tokens"] for s in samples]
    report_tokens = torch.stack([s["report_tokens"] for s in samples])
    return {
        "images": images,
        "annotations_tokens": annotations_tokens,
        "report_tokens": report_tokens
    }

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
        outputs = self.fc(outputs)  # (batch, seq_length, vocab_size)
        return outputs

    def generate(self, features, max_length=100):
        batch_size = features.size(0)
        h0 = features.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        input_token = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=features.device)
        generated_tokens = []
        hidden = (h0, c0)
        for _ in range(max_length):
            embedded = self.embedding(input_token)
            output, hidden = self.lstm(embedded, hidden)
            output = self.fc(output)  # (batch, 1, vocab_size)
            token = output.argmax(dim=-1)  # (batch, 1)
            generated_tokens.append(token)
            input_token = token
            if (token == tokenizer.eos_token_id).all():
                break
        generated_tokens = torch.cat(generated_tokens, dim=1)  # (batch, seq_length)
        return generated_tokens

# 4. Multi-Modal Model: Fusing image and annotation features
class MultiModalModel(nn.Module):
    def __init__(self, vocab_size, use_radimagenet=False, rad_weights_path=None):
        super(MultiModalModel, self).__init__()
        self.image_encoder = ImageEncoder(use_radimagenet, rad_weights_path)
        self.annotation_encoder = AnnotationEncoder(vocab_size)
        # Concatenate image features (1024-d) and annotation features (256-d) → 1280-d vector.
        # Average these vectors over all images in the study and map to 256-d.
        self.fusion_fc = nn.Linear(1280, 256)
        self.decoder = ReportDecoder(vocab_size)
    
    def forward(self, images, annotations_tokens, report_tokens):
        pair_features = []
        for img, ann_tokens in zip(images, annotations_tokens):
            img_feat = self.image_encoder(img.unsqueeze(0))  # (1, 1024)
            ann_feat = self.annotation_encoder(ann_tokens.unsqueeze(0))  # (1, 256)
            pair_feat = torch.cat([img_feat, ann_feat], dim=1)  # (1, 1280)
            pair_features.append(pair_feat)
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
# Training Pipeline
############################################

def train_model(root_dir, num_epochs=10, batch_size=1, lr=1e-4, use_radimagenet=False, rad_weights_path=None):
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Initialize GPT-2 tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<SOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_toks} special tokens. Vocabulary size: {tokenizer.vocab_size}")

    # Create dataset and dataloader.
    dataset = MultiModalStudyDataset(root_dir, image_transform, tokenizer, max_text_length=300)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Instantiate multi-modal model.
    model = MultiModalModel(tokenizer.vocab_size, use_radimagenet, rad_weights_path).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_iterations = len(dataloader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        for i, batch in enumerate(dataloader):
            # Since batch_size=1, extract the single study.
            images = batch["images"][0]  # List of image tensors.
            annotations_tokens = batch["annotations_tokens"][0]  # List of token tensors.
            report_tokens = batch["report_tokens"].to(device)  # (1, seq_length)

            optimizer.zero_grad()
            # Teacher forcing: input report_tokens[:, :-1], target report_tokens[:, 1:].
            outputs = model(images, annotations_tokens, report_tokens[:, :-1])
            targets = report_tokens[:, 1:]
            outputs = outputs.reshape(-1, tokenizer.vocab_size)
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            iter_time = time.time() - epoch_start
            remaining_iters = total_iterations - (i + 1)
            eta = remaining_iters * (iter_time / (i + 1))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            logger.info(f"Epoch {epoch+1}, Iteration {i+1}/{total_iterations}, Loss: {loss.item():.4f}, ETA: {eta_str}")
        
        avg_loss = epoch_loss / total_iterations
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint.
        ckpt_path = os.path.join("checkpoints", f"checkpoint_epoch_{epoch+1}.pt")
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint = {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # Save the tokenizer vocabulary from GPT2Tokenizer.
            "tokenizer_vocab": tokenizer.get_vocab()
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    root_folder = "pilot_dataset"  # Assumed to be in project root.
    rad_weights_path = "weights/radimagenet_densenet121.pth"  # Set as needed.
    train_model(root_folder, num_epochs=10, batch_size=1, lr=1e-4, use_radimagenet=False, rad_weights_path=rad_weights_path)
