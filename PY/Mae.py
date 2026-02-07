import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SFM_CSV = "sfm_features.csv" 
BATCH_SIZE = 8
LEARNING_RATE = 2e-5 
EPOCHS = 15

print(f"🚀 Running on: {DEVICE}")

# --- DATASET ---
class VoicePathologyDataset(Dataset):
    def __init__(self, df, sfm_scaler, sfm_cols):
        self.df = df.reset_index(drop=True)
        self.sfm_scaler = sfm_scaler
        self.sfm_cols = sfm_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['file_path']
        
        # Physics Features
        raw_sfm = row[self.sfm_cols].values.astype(np.float32)
        norm_sfm = self.sfm_scaler.transform([raw_sfm])[0]
        sfm_tensor = torch.tensor(norm_sfm, dtype=torch.float32)

        # Label
        label = torch.tensor(row['label_encoded'], dtype=torch.long)

        # Return path (string), sfm (tensor), label (tensor)
        return file_path, sfm_tensor, label

# --- MODEL ---
class DualStreamModel(nn.Module):
    def __init__(self, num_classes, sfm_dim=10, fusion_dim=512):
        super().__init__()
        
        print("🧠 Loading Hance-AI AudioMAE...")
        self.audio_encoder = AutoModel.from_pretrained(
            "hance-ai/audiomae", 
            trust_remote_code=True
        )
        self.audio_hidden_size = 768 
        self.audio_proj = nn.Linear(self.audio_hidden_size, fusion_dim)

        # Physics Stream
        self.sfm_mlp = nn.Sequential(
            nn.Linear(sfm_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # Fusion
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio_paths, sfm_input):
        # 1. Audio Stream
        audio_feats_list = []
        
        for path in audio_paths:
            # Encoder returns (768, 8, 64)
            # We force the model to look at 'path'
            feat = self.audio_encoder(path) 
            audio_feats_list.append(feat)
            
        # Stack into batch -> (Batch, 768, 8, 64)
        audio_feats = torch.stack(audio_feats_list)
        
        # --- CRITICAL FIX: FORCE TO GPU ---
        # Ensure the stacked tensor matches the device of the Linear layers
        device = self.audio_proj.weight.device
        audio_feats = audio_feats.to(device)
        
        # Reshape: (Batch, 768, 8, 64) -> (Batch, 768, 512)
        batch_size = audio_feats.shape[0]
        audio_feats = audio_feats.view(batch_size, 768, -1)
        
        # Permute for Transformer: (Batch, 512, 768)
        audio_feats = audio_feats.permute(0, 2, 1)
        
        # Project to 512 dim (Now this will work because both are on GPU)
        audio_emb = self.audio_proj(audio_feats) 

        # 2. Physics Stream
        sfm_emb = self.sfm_mlp(sfm_input) 
        sfm_query = sfm_emb.unsqueeze(1) 

        # 3. Fusion
        attn_out, attn_weights = self.cross_attn(
            query=sfm_query, 
            key=audio_emb, 
            value=audio_emb
        )
        
        fused_vector = attn_out.squeeze(1)
        logits = self.classifier(fused_vector)
        
        return logits, attn_weights
# --- SETUP ---
full_df = pd.read_csv(SFM_CSV)
label_encoder = LabelEncoder()
full_df['label_encoded'] = label_encoder.fit_transform(full_df['label_name'])
NUM_CLASSES = len(label_encoder.classes_)

sfm_cols = ['jitter_local', 'jitter_rap', 'shimmer_local', 'shimmer_apq3', 
            'hnr', 'f1', 'f2', 'f3', 'f4', 'f0_mean']
scaler = StandardScaler()
train_subset = full_df[full_df['split'] == 'train']
scaler.fit(train_subset[sfm_cols].values)

def custom_collate(batch):
    paths = [item[0] for item in batch] 
    sfms = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return paths, sfms, labels

train_ds = VoicePathologyDataset(full_df[full_df['split'] == 'train'], scaler, sfm_cols)
val_ds = VoicePathologyDataset(full_df[full_df['split'] == 'val'], scaler, sfm_cols)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

# --- TRAIN ---
model = DualStreamModel(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("🔥 Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for paths, sfms, labels in train_loader:
        sfms, labels = sfms.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits, _ = model(paths, sfms)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")