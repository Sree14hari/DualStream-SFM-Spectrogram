

# ==========================================
# 1. UPGRADED HYBRID DATASET
# ==========================================
class HybridDatasetV2(Dataset):
    def __init__(self, df, sfm_scaler, sfm_cols, is_train=False):
        self.df = df.reset_index(drop=True)
        self.sfm_scaler = sfm_scaler
        self.sfm_cols = sfm_cols
        self.is_train = is_train
        
        # Audio Transforms
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, 
            n_fft=1024, 
            n_mels=128,
            hop_length=512
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        
        # Image Transforms with Augmentation
        self.resize = T.Resize((224, 224), antialias=True)
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Training Augmentations
        if self.is_train:
            self.augment = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomErasing(p=0.3, scale=(0.02, 0.2))
            ])

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # A. LOAD AUDIO (ResNet Input)
        try: 
            wav, sr = torchaudio.load(row['file_path'])
        except: 
            wav = torch.zeros(1, 16000*3)
            sr = 16000
            
        if sr != 16000: 
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        
        # Pad or trim to 3 seconds
        target_length = 16000 * 3
        if wav.shape[1] < target_length:
            wav = F.pad(wav, (0, target_length - wav.shape[1]))
        else:
            wav = wav[:, :target_length]
        
        # Generate Spectrogram Image
        img = self.resize(self.db(self.mel(wav))).repeat(3, 1, 1)  # 1ch -> 3ch
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # MinMax Scale
        
        # Apply augmentation if training
        if self.is_train:
            img = self.augment(img)
            
        img = self.norm(img)  # ImageNet Norm
        
        # B. LOAD PHYSICS (Dual-Stream Input)
        raw_sfm = row[self.sfm_cols].values.astype(np.float32)
        norm_sfm = self.sfm_scaler.transform([raw_sfm])[0]
        
        return (
            img, 
            torch.tensor(norm_sfm, dtype=torch.float32), 
            torch.tensor(row['label_encoded'], dtype=torch.long)
        )

# ==========================================
# 2. ATTENTION FUSION MODULE
# ==========================================
class AttentionFusion(nn.Module):
    """Learns to weight vision vs physics features dynamically"""
    def __init__(self, vis_dim, phys_dim):
        super().__init__()
        self.vis_attn = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 4),
            nn.ReLU(),
            nn.Linear(vis_dim // 4, 1),
            nn.Sigmoid()
        )
        self.phys_attn = nn.Sequential(
            nn.Linear(phys_dim, phys_dim // 4),
            nn.ReLU(),
            nn.Linear(phys_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vis_feat, phys_feat):
        # Calculate attention weights
        vis_weight = self.vis_attn(vis_feat)
        phys_weight = self.phys_attn(phys_feat)
        
        # Normalize weights to sum to 1
        total_weight = vis_weight + phys_weight
        vis_weight = vis_weight / (total_weight + 1e-6)
        phys_weight = phys_weight / (total_weight + 1e-6)
        
        # Apply attention and concatenate
        weighted_vis = vis_feat * vis_weight
        weighted_phys = phys_feat * phys_weight
        
        return torch.cat([weighted_vis, weighted_phys], dim=1)

# ==========================================
# 3. UPGRADED HYBRID MODEL
# ==========================================
class HybridResNet50Upgraded(nn.Module):
    def __init__(self, num_classes, physics_dim=10):
        super().__init__()
        
        # Branch 1: ResNet-50 (Vision) - UPGRADED FROM ResNet-34
        print("    Loading Pre-trained ResNet-50...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer to get features
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_dim = 2048  # ResNet-50 output dimension
        
        # Branch 2: DEEPER Physics MLP
        self.physics_mlp = nn.Sequential(
            nn.Linear(physics_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.physics_out_dim = 128
        
        # Attention-based Fusion
        self.fusion = AttentionFusion(self.resnet_dim, self.physics_out_dim)
        fusion_dim = self.resnet_dim + self.physics_out_dim  # 2048 + 128 = 2176
        
        # Residual Gate (allows model to bypass physics if needed)
        self.residual_gate = nn.Sequential(
            nn.Linear(self.physics_out_dim, 1),
            nn.Sigmoid()
        )
        
        # Enhanced Classifier with Residual Connection
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, num_classes)
        )

    def forward(self, images, physics):
        # 1. Vision Path
        vis_feat = self.resnet_features(images)  # [Batch, 2048, 1, 1]
        vis_feat = vis_feat.view(vis_feat.size(0), -1)  # Flatten -> [Batch, 2048]
        
        # 2. Physics Path
        phys_feat = self.physics_mlp(physics)  # [Batch, 128]
        
        # 3. Attention-based Fusion
        combined = self.fusion(vis_feat, phys_feat)  # [Batch, 2176]
        
        # 4. Classification
        return self.classifier(combined)

# ==========================================
# 4. DATA PREPARATION
# ==========================================
if not os.path.exists(SFM_CSV):
    print(f"❌ ERROR: '{SFM_CSV}' not found. Please upload dataset.")
    exit()

full_df = pd.read_csv(SFM_CSV)
le = LabelEncoder()
full_df['label_encoded'] = le.fit_transform(full_df['label_name'])
NUM_CLASSES = len(le.classes_)
print(f"📋 Classes Detected: {le.classes_}")

sfm_cols = ['jitter_local', 'jitter_rap', 'shimmer_local', 'shimmer_apq3', 
            'hnr', 'f1', 'f2', 'f3', 'f4', 'f0_mean']

# Fit scaler on training data only
scaler = StandardScaler().fit(full_df[full_df['split'] == 'train'][sfm_cols].values)

# Initialize Datasets with augmentation flag
train_ds = HybridDatasetV2(full_df[full_df['split'] == 'train'], scaler, sfm_cols, is_train=True)
val_ds = HybridDatasetV2(full_df[full_df['split'] == 'val'], scaler, sfm_cols, is_train=False)
test_ds = HybridDatasetV2(full_df[full_df['split'] == 'test'], scaler, sfm_cols, is_train=False)

# Initialize Loaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ==========================================
# 5. MODEL INITIALIZATION WITH DIFFERENTIAL LR
# ==========================================
model = HybridResNet50Upgraded(NUM_CLASSES).to(DEVICE)

# Differential Learning Rates
optimizer = torch.optim.AdamW([
    {'params': model.resnet_features.parameters(), 'lr': 1e-5},  # Slow for pretrained
    {'params': model.physics_mlp.parameters(), 'lr': 5e-4},      # Fast for physics
    {'params': model.fusion.parameters(), 'lr': 1e-4},           # Medium for fusion
    {'params': model.classifier.parameters(), 'lr': 1e-4}        # Medium for classifier
], weight_decay=1e-4)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

criterion = nn.CrossEntropyLoss()

# ==========================================
# 6. TRAINING LOOP WITH WARMUP
# ==========================================
print("\n🔥 Starting UPGRADED Hybrid Training with Warmup...")
best_acc = 0.0
train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    # WARMUP: Freeze ResNet for first few epochs
    if epoch < WARMUP_EPOCHS:
        print(f"❄️  WARMUP MODE: ResNet Frozen")
        for param in model.resnet_features.parameters():
            param.requires_grad = False
    elif epoch == WARMUP_EPOCHS:
        print(f"🔓 UNFREEZING ResNet")
        for param in model.resnet_features.parameters():
            param.requires_grad = True
    
    # TRAINING PHASE
    model.train()
    total_loss = 0
    
    for imgs, phys, labels in train_loader:
        imgs, phys, labels = imgs.to(DEVICE), phys.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(imgs, phys)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # VALIDATION PHASE
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, phys, labels in val_loader:
            imgs, phys = imgs.to(DEVICE), phys.to(DEVICE)
            logits = model(imgs, phys)
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_targets.extend(labels.numpy())
    
    val_acc = accuracy_score(val_targets, val_preds)
    val_accuracies.append(val_acc)
    
    # Update learning rate based on validation accuracy
    scheduler.step(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, "best_hybrid_resnet50_upgraded.pth")
        print(f"    💾 Saved Best Model ({val_acc*100:.2f}%)")

# ==========================================
# 7. FINAL EVALUATION
# ==========================================
print("\n🚀 Evaluating Best Model on Test Set...")
checkpoint = torch.load("best_hybrid_resnet50_upgraded.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds, all_labels = [], []
all_probs = []

with torch.no_grad():
    for imgs, phys, labels in test_loader:
        imgs, phys = imgs.to(DEVICE), phys.to(DEVICE)
        logits = model(imgs, phys)
        probs = F.softmax(logits, dim=1)
        
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# Metrics
test_acc = accuracy_score(all_labels, all_preds)
print(f"\n{'='*60}")
print(f"🏆 FINAL HYBRID ACCURACY: {test_acc*100:.2f}%")
print(f"{'='*60}")
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_, digits=4))

# ==========================================
# 8. VISUALIZATIONS
# ==========================================

# A. Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='magma', 
            xticklabels=le.classes_, yticklabels=le.classes_,
            cbar_kws={'label': 'Count'})
plt.title(f'Upgraded Hybrid ResNet-50 Confusion Matrix\nAccuracy: {test_acc*100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_upgraded.png', dpi=300, bbox_inches='tight')
plt.show()

# B. Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss curve
ax1.plot(train_losses, label='Training Loss', linewidth=2, color='#e74c3c')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy curve
ax2.plot(val_accuracies, label='Validation Accuracy', linewidth=2, color='#2ecc71')
ax2.axhline(y=0.96, color='r', linestyle='--', label='Target (96%)', alpha=0.7)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_upgraded.png', dpi=300, bbox_inches='tight')
plt.show()

# C. Physics Feature Importance
print("\n🔍 Analyzing Physics Feature Importance...")
weights = model.physics_mlp[0].weight.detach().cpu().numpy()
importance = np.mean(np.abs(weights), axis=0)

plt.figure(figsize=(12, 6))
colors = sns.color_palette("viridis", len(sfm_cols))
bars = plt.barh(sfm_cols, importance, color=colors)
plt.title("Clinical Physics Feature Importance\n(First Layer Weights)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Mean Absolute Weight", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, importance)):
    plt.text(val, i, f' {val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance_upgraded.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 9. PER-CLASS ACCURACY ANALYSIS
# ==========================================
print("\n📈 Per-Class Accuracy Analysis:")
print("-" * 60)
for i, class_name in enumerate(le.classes_):
    class_mask = np.array(all_labels) == i
    if class_mask.sum() > 0:
        class_acc = accuracy_score(
            np.array(all_labels)[class_mask], 
            np.array(all_preds)[class_mask]
        )
        print(f"{class_name:20s}: {class_acc*100:6.2f}% ({class_mask.sum()} samples)")

print("\n✅ Training Complete! Best validation accuracy: {:.2f}%".format(best_acc * 100))