# transformer_de.py - Transformer-based Direction Estimator with Explicit Attention Analysis
import os
import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, List

DEFAULT_BERT_NAME = os.environ.get('SDM_BERT_MODEL', 'xlm-roberta-base')
ENV_USE_FOCAL = os.environ.get('SDM_USE_FOCAL', '0') == '1'
ENV_FOCAL_GAMMA = float(os.environ.get('SDM_FOCAL_GAMMA', '2.0'))
ENV_EARLY_STOP = int(os.environ.get('SDM_EARLY_STOP_PATIENCE', '5'))
ENV_CLIP_MAX_NORM = float(os.environ.get('SDM_CLIP_MAX_NORM', '1.0'))

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ 未安装 transformers，运行: pip install transformers")

def _has_new_torch_amp():
    return hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

_HAS_NEW_AMP = _has_new_torch_amp()

def _autocast_ctx(enabled=True):
    if not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast('cuda', enabled=True) if _HAS_NEW_AMP else torch.cuda.amp.autocast(enabled=True)

def _make_scaler(enabled=True):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler('cuda', enabled=True) if _HAS_NEW_AMP else torch.cuda.amp.GradScaler(enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, target):
        logpt = torch.log_softmax(logits, dim=-1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.view(-1, 1)).squeeze(1)
        pt = pt.gather(1, target.view(-1, 1)).squeeze(1)
        if self.weight is not None:
            w = self.weight.to(logits.device).gather(0, target)
            loss = -w * (1 - pt).pow(self.gamma) * logpt
        else:
            loss = -(1 - pt).pow(self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None, 
                key_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = True):
        attn_output, attn_weights = self.self_attn(
            x, x, x, 
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        if return_attention:
            return x, attn_weights
        return x

class AttentionDirectionAnalyzer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.direction_scorer = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3)
        )
    
    def forward(self, attention_weights: List[torch.Tensor], 
                input_ids: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = attention_weights[0].size(0)
        device = attention_weights[0].device
        
        avg_attn = torch.stack(attention_weights, dim=0).mean(dim=0)
        avg_attn = avg_attn.mean(dim=1)
        
        seq_len = avg_attn.size(1)
        forward_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        backward_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
        
        mask_2d = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
        
        forward_attn = (avg_attn * forward_mask * mask_2d).sum(dim=-1)
        backward_attn = (avg_attn * backward_mask * mask_2d).sum(dim=-1)
        
        valid_forward = (forward_mask * mask_2d).sum(dim=-1) + 1e-8
        valid_backward = (backward_mask * mask_2d).sum(dim=-1) + 1e-8
        
        forward_attn = forward_attn / valid_forward
        backward_attn = backward_attn / valid_backward
        
        valid_len = padding_mask.sum(dim=1, keepdim=True)
        forward_strength = (forward_attn * padding_mask).sum(dim=1) / valid_len.squeeze()
        backward_strength = (backward_attn * padding_mask).sum(dim=1) / valid_len.squeeze()
        
        ratio = forward_strength / (backward_strength + 1e-8)
        diff = forward_strength - backward_strength
        balance = 1.0 / (torch.abs(diff) + 1e-2)
        
        attn_var_forward = ((forward_attn - forward_strength.unsqueeze(1)) ** 2 * padding_mask).sum(dim=1) / valid_len.squeeze()
        attn_var_backward = ((backward_attn - backward_strength.unsqueeze(1)) ** 2 * padding_mask).sum(dim=1) / valid_len.squeeze()
        
        features = torch.stack([
            forward_strength,
            backward_strength,
            ratio,
            diff,
            balance,
            (attn_var_forward + attn_var_backward) / 2
        ], dim=-1)
        
        direction_logits = self.direction_scorer(features)
        
        return direction_logits

class TransformerDirectionEstimator(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int = 4, 
                 num_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.5, freeze_embedding: bool = False, 
                 bert_model_name: str = DEFAULT_BERT_NAME,
                 use_explicit_attention: bool = True,
                 explicit_weight: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.freeze_embedding = freeze_embedding
        self.bert_model_name = bert_model_name
        self.dropout = dropout
        self.use_explicit_attention = use_explicit_attention
        self.explicit_weight = explicit_weight
        
        if HAS_TRANSFORMERS:
            print(f"🚀 加载预训练模型并提取 embedding: {bert_model_name}")
            try:
                cfg = AutoConfig.from_pretrained(bert_model_name)
                backbone = AutoModel.from_pretrained(bert_model_name, config=cfg, low_cpu_mem_usage=True)
            except Exception:
                backbone = AutoModel.from_pretrained(bert_model_name)
            
            bert_dim = backbone.config.hidden_size
            self.embedding = backbone.get_input_embeddings()
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.pad_token_id = tokenizer.pad_token_id
            print(f"✓ Pad Token ID: {self.pad_token_id}")
            
            if freeze_embedding:
                print("❄️ 冻结 embedding 权重")
                for p in self.embedding.parameters():
                    p.requires_grad = False
            else:
                print("🔥 微调 embedding 权重")
            
            if bert_dim != d_model:
                self.projection = nn.Linear(bert_dim, d_model)
                print(f"📐 添加投影层: {bert_dim} -> {d_model}")
            else:
                self.projection = None
            print("✓ 使用预训练 embedding")
        else:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.projection = None
            self.pad_token_id = 0
            print("⚠️ 使用随机初始化 embedding")
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        self.implicit_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        
        if use_explicit_attention:
            self.attention_analyzer = AttentionDirectionAnalyzer(d_model)
            print(f"✓ 启用显式Attention分析 (权重={explicit_weight})")
        else:
            self.attention_analyzer = None
        
        print(f"✓ Transformer配置: {num_layers}层, {nhead}头, d_model={d_model}")
    
    def _infer_mask(self, input_ids):
        pad_token_id = getattr(self, 'pad_token_id', 0)
        return (input_ids != pad_token_id).float()
    
    def forward(self, input_ids):
        if self.projection is not None:
            x = self.projection(self.embedding(input_ids))
        else:
            x = self.embedding(input_ids)
        
        x = self.pos_encoder(x)
        
        padding_mask = self._infer_mask(input_ids)
        key_padding_mask = (padding_mask == 0)
        
        all_attentions = []
        return_attn = self.use_explicit_attention and self.attention_analyzer is not None
        
        for layer in self.encoder_layers:
            if return_attn:
                x, attn_weights = layer(x, key_padding_mask=key_padding_mask, return_attention=True)
                all_attentions.append(attn_weights)
            else:
                x = layer(x, key_padding_mask=key_padding_mask, return_attention=False)
        
        cls_output = x[:, 0, :]
        
        attn_weights = self.attention_pool(x)
        attn_weights = attn_weights * padding_mask.unsqueeze(-1)
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)
        attn_output = (x * attn_weights).sum(dim=1)
        
        combined = torch.cat([cls_output, attn_output], dim=-1)
        implicit_logits = self.implicit_classifier(combined)
        
        if return_attn and len(all_attentions) > 0:
            explicit_logits = self.attention_analyzer(all_attentions, input_ids, padding_mask)
            alpha = self.explicit_weight
            logits = (1 - alpha) * implicit_logits + alpha * explicit_logits
        else:
            logits = implicit_logits
        
        return logits

class DirectionDatasetBERT(Dataset):
    def __init__(self, csv_path: str, max_length: int = 64,
                 use_bert_tokenizer: bool = True, bert_model_name: str = DEFAULT_BERT_NAME):
        self.max_length = max_length
        self.use_bert_tokenizer = use_bert_tokenizer
        
        df = pd.read_csv(csv_path)
        if 'text' not in df.columns or 'direction' not in df.columns:
            raise ValueError("CSV必须包含 'text' 和 'direction' 列")
        
        self.texts = df['text'].astype(str).tolist()
        
        label_map = {
            'left': 0, 'right': 1, 'bidirectional': 2,
            'forward': 0, 'backward': 1, 'mutual': 2,
            '左': 0, '右': 1, '双向': 2,
            '正向': 0, '反向': 1, '互为': 2,
            '正向因果': 0, '反向因果': 1, '双向因果': 2,
            'l': 0, 'r': 1, 'b': 2,
            'L': 0, 'R': 1, 'B': 2,
            'f': 0, 'bw': 1, 'm': 2,
            '0': 0, '1': 1, '2': 2,
            0: 0, 1: 1, 2: 2
        }
        
        raw_directions = df['direction'].tolist()
        self.directions = []
        for d in raw_directions:
            if d in label_map:
                self.directions.append(label_map[d])
            elif str(d).strip() in label_map:
                self.directions.append(label_map[str(d).strip()])
            else:
                try:
                    self.directions.append(int(d))
                except:
                    raise ValueError(f"无效标签: '{d}'")
        
        if use_bert_tokenizer:
            if not HAS_TRANSFORMERS:
                raise RuntimeError("需要 transformers")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.vocab_size = self.tokenizer.vocab_size
            print(f"✓ BERT Tokenizer: {bert_model_name}, vocab={self.vocab_size}")
        else:
            self.char_to_idx = self._build_vocab(self.texts)
            self.vocab_size = len(self.char_to_idx)
            print(f"✓ Char Tokenizer, vocab={self.vocab_size}")
    
    def _build_vocab(self, texts):
        chars = set()
        for t in texts:
            chars.update(t)
        char_to_idx = {c: i+1 for i, c in enumerate(sorted(chars))}
        char_to_idx['<PAD>'] = 0
        return char_to_idx
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        direction = self.directions[idx]
        
        if self.use_bert_tokenizer:
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
        else:
            tokens = [self.char_to_idx.get(ch, 1) for ch in text[:self.max_length]]
            if len(tokens) < self.max_length:
                tokens += [0] * (self.max_length - len(tokens))
            input_ids = torch.tensor(tokens, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'direction': torch.tensor(direction, dtype=torch.long)
        }

def _calc_class_weights(dataset):
    if isinstance(dataset, Subset):
        labels = [dataset.dataset.directions[i] for i in dataset.indices]
    else:
        labels = dataset.directions
    
    counts = np.bincount(labels, minlength=3)
    total = len(labels)
    weights = total / (3.0 * counts + 1e-8)
    return torch.FloatTensor(weights)

def _per_class_metrics(cm, eps=1e-12):
    P, R, F = [], [], []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        p = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r+eps) if (p+r) > 0 else 0.0
        P.append(p); R.append(r); F.append(f1)
    return np.array(P), np.array(R), np.array(F), float(np.mean(P)), float(np.mean(R)), float(np.mean(F))

def train_transformer_de(
    model,
    train_loader,
    val_loader,
    num_epochs=30,
    lr=1e-3,
    device='cuda',
    save_path='best_model.pth',
    use_focal=False,
    focal_gamma=2.0,
    early_stop_patience=5,
    clip_max_norm=1.0,
    extra_meta=None
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    class_weights = _calc_class_weights(train_loader.dataset)
    print(f"📊 类别权重: {class_weights.tolist()}")
    
    if use_focal:
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
        print(f"🎯 Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("🎯 CrossEntropy Loss")
    
    use_amp = (device == 'cuda' and os.environ.get('SDM_NO_AMP') != '1')
    scaler = _make_scaler(enabled=use_amp) if use_amp else None
    print(f"⚡ AMP: {'启用' if use_amp else '禁用'}")
    
    direction_names = ['正向因果(A→B)', '反向因果(A←B)', '双向因果(A↔B)']
    
    print("="*60)
    print("训练 Transformer DE (显式Attention分析)")
    print("="*60)
    
    best_val_acc = 0.0
    no_improve = 0
    patience = early_stop_patience
    
    for epoch in range(num_epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['direction'].to(device)
            
            optimizer.zero_grad()
            
            if use_amp and scaler is not None:
                with _autocast_ctx(enabled=True):
                    logits = model(input_ids)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                if clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids)
                loss = criterion(logits, labels)
                loss.backward()
                if clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                optimizer.step()
            
            tr_loss += float(loss.item())
            preds = torch.argmax(logits.detach(), dim=-1)
            tr_correct += int((preds == labels).sum().item())
            tr_total += labels.size(0)
        
        avg_train_loss = tr_loss / max(1, len(train_loader))
        train_acc = 100.0 * tr_correct / max(1, tr_total)
        
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        cm = np.zeros((3, 3), dtype=int)
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['direction'].to(device)
                with _autocast_ctx(enabled=use_amp):
                    logits = model(input_ids)
                    loss = criterion(logits, labels)
                va_loss += float(loss.item())
                preds = torch.argmax(logits, dim=-1)
                va_correct += int((preds == labels).sum().item())
                va_total += labels.size(0)
                for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    cm[t][p] += 1
        
        avg_val_loss = va_loss / max(1, len(val_loader))
        val_acc = 100.0 * va_correct / max(1, va_total)
        P, R, F1, macro_p, macro_r, macro_f1 = _per_class_metrics(cm)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%  | Macro P/R/F1: {macro_p:.3f}/{macro_r:.3f}/{macro_f1:.3f}')
        
        if (epoch + 1) % 5 == 0:
            print('\n混淆矩阵:')
            header = '实际\\预测'
            print(f'{header:15s}  ' + '  '.join([f'{n:18s}' for n in direction_names]))
            for i, row in enumerate(cm):
                print(f'{direction_names[i]:18s}  ' + '  '.join([f'{v:18d}' for v in row]))
            print('按类 P/R/F1:')
            for i, name in enumerate(direction_names):
                print(f'  {name}: P={P[i]:.3f} R={R[i]:.3f} F1={F1[i]:.3f}')
            print()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            
            ds_flag = True
            char_to_idx = None
            try:
                ds = train_loader.dataset
                real_ds = ds.dataset if isinstance(ds, Subset) else ds
                ds_flag = getattr(real_ds, 'use_bert_tokenizer', True)
                if not ds_flag:
                    char_to_idx = getattr(real_ds, 'char_to_idx', None)
            except Exception:
                pass
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'vocab_size': getattr(model.embedding, 'num_embeddings', None) or 0,
                'd_model': model.d_model,
                'nhead': model.nhead,
                'num_layers': model.num_layers,
                'use_bert_tokenizer': ds_flag,
                'freeze_embedding': model.freeze_embedding,
                'bert_model_name': getattr(model, 'bert_model_name', DEFAULT_BERT_NAME),
                'pad_token_id': getattr(model, 'pad_token_id', 0),
                'class_weights': class_weights.detach().cpu().tolist(),
                'use_focal': use_focal,
                'focal_gamma': focal_gamma,
                'dropout': getattr(model, 'dropout', 0.5),
                'use_explicit_attention': getattr(model, 'use_explicit_attention', True),
                'explicit_weight': getattr(model, 'explicit_weight', 0.3),
            }
            if char_to_idx is not None:
                save_dict['char_to_idx'] = char_to_idx
            if extra_meta:
                save_dict.update(extra_meta)
            
            torch.save(save_dict, save_path)
            print(f'  ✓ 保存 -> {save_path} (验证准确率: {val_acc:.2f}%)\n')
        else:
            no_improve += 1
            print(f'  ↳ 未提升 ({no_improve}/{patience})\n')
            if no_improve >= patience:
                print(f'⏹ EarlyStopping')
                break
        
        scheduler.step()
    
    print(f'\n训练完成！最佳验证准确率: {best_val_acc:.2f}%')
    return model, device

def _build_cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default='training_data.csv')
    p.add_argument('--bert-model', type=str, default=DEFAULT_BERT_NAME)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dim-feedforward', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--freeze-embedding', action='store_true')
    p.add_argument('--max-length', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--train-split', type=float, default=0.8)
    p.add_argument('--use-focal', action='store_true')
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--early-stop', type=int, default=5)
    p.add_argument('--clip-max-norm', type=float, default=1.0)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--save-name', type=str, default='best_model_transformer.pth')
    p.add_argument('--no-explicit-attention', action='store_true')
    p.add_argument('--explicit-weight', type=float, default=0.3)
    return p

def main():
    print("="*60)
    print("Transformer DE (显式Attention分析)")
    print("="*60)
    
    if not HAS_TRANSFORMERS:
        print("\n❌ 需要 transformers")
        return
    
    args = _build_cli().parse_args()
    
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.no_amp:
        os.environ['SDM_NO_AMP'] = '1'
    
    dataset = DirectionDatasetBERT(
        csv_path=args.csv,
        max_length=args.max_length,
        use_bert_tokenizer=True,
        bert_model_name=args.bert_model
    )
    
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"✓ 训练: {train_size} | 验证: {val_size}")
    
    model = TransformerDirectionEstimator(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        freeze_embedding=args.freeze_embedding,
        bert_model_name=args.bert_model,
        use_explicit_attention=(not args.no_explicit_attention),
        explicit_weight=args.explicit_weight
    )
    
    save_path = os.path.join('checkpoints', args.save_name)
    train_transformer_de(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=dev,
        save_path=save_path,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        early_stop_patience=args.early_stop,
        clip_max_norm=args.clip_max_norm,
        extra_meta={'max_length': args.max_length, 'config': vars(args)}
    )
    
    print(f"\n✅ 完成: {save_path}")

if __name__ == '__main__':
    main()