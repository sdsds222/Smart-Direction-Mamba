# mamba_de.py (FIXED VERSION - ALL BUGS RESOLVED, INTERFACES UNCHANGED)
import os
import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import numpy as np
from typing import Optional, Union

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

class SimplifiedMambaSSM(nn.Module):
    """
    修正点：
    1) 将 A 改为对角参数 A_diag，离散化使用 exp(Δ * A_diag) 与 h 做按维缩放，避免错误的逐元素矩阵指数。
    2) 仍保持与原类名/接口一致。
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 使用对角 A：保证 e^{ΔA} = diag(e^{Δ a_i}) 简单稳定
        self.A_diag = nn.Parameter(torch.randn(d_state) * 0.01)

        self.delta_proj = nn.Linear(d_model, d_state, bias=False)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.input_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # 稳定性：A 对角为负
        with torch.no_grad():
            self.A_diag.data = -self.A_diag.data.abs()

    def forward(self, x, h=None, mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        device = x.device
        if h is None:
            h = torch.zeros(B, self.d_state, device=device)
        if mask is None:
            mask = torch.ones(B, T, device=device)
        mask = mask.unsqueeze(-1)  # [B, T, 1]

        outs = []
        for t in range(T):
            x_t = x[:, t, :]
            m_t = mask[:, t, :]

            x_gated = torch.sigmoid(self.input_gate(x_t)) * x_t
            delta = F.softplus(self.delta_proj(x_gated))          # [B, d_state]
            B_t = self.B_proj(x_gated)                            # [B, d_state]
            C_t = self.C_proj(x_gated)                            # [B, d_state]

            # e^{Δ * A_diag}：按维缩放
            A_bar = torch.exp(delta * self.A_diag)                # [B, d_state]
            h_candidate = torch.tanh(A_bar * h + B_t)             # [B, d_state]
            h = h_candidate * m_t + h * (1.0 - m_t)

            s_t = C_t * h                                         # [B, d_state]
            y_t = self.out(s_t) + self.D * x_t                    # [B, d_model]
            outs.append(y_t)

        return torch.stack(outs, dim=1), h  # [B, T, d_model], [B, d_state]

class MambaDirectionEstimator(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_state: int = 16, dropout: float = 0.5,
                 freeze_embedding: bool = False, bert_model_name: str = DEFAULT_BERT_NAME):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.freeze_embedding = freeze_embedding
        self.bert_model_name = bert_model_name
        self.dropout = dropout

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
            print("⚠️ 使用随机初始化 embedding（未安装 transformers）")

        self.forward_ssm = SimplifiedMambaSSM(d_model, d_state)
        self.backward_ssm = SimplifiedMambaSSM(d_model, d_state)

        self.state_analyzer = nn.Sequential(
            nn.Linear(d_state * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.pool_proj = nn.Linear(d_model * 2, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )

    def _infer_mask(self, input_ids):
        pad_token_id = getattr(self, 'pad_token_id', 0)
        return (input_ids != pad_token_id).float()

    def forward(self, input_ids):
        # [B, T] -> [B, T, d_model]
        if self.projection is not None:
            x = self.projection(self.embedding(input_ids))
        else:
            x = self.embedding(input_ids)

        # 推断有效位 mask
        mask = self._infer_mask(input_ids)             # [B, T]

        # 前向与反向 SSM
        forward_out, h_f = self.forward_ssm(x, mask=mask)  # [B, T, d], [B, s]
        x_reversed = torch.flip(x, dims=[1])
        mask_reversed = torch.flip(mask, dims=[1])
        backward_out, h_b = self.backward_ssm(x_reversed, mask=mask_reversed)
        backward_out = torch.flip(backward_out, dims=[1])  # 对齐时间

        # 拼接双向输出
        combined = torch.cat([forward_out, backward_out], dim=-1)  # [B, T, 2d]

        # ---- 关键修复：带 mask 的池化，避免 PAD 稀释 ----
        mask_exp = mask.unsqueeze(-1)                               # [B, T, 1]
        combined_masked = combined * mask_exp                       # [B, T, 2d]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)       # [B, 1]
        pooled_seq = combined_masked.sum(dim=1) / denom             # [B, 2d]
        pooled = self.pool_proj(pooled_seq)                         # [B, d]
        # ---------------------------------------------------

        # 融合终态
        final_state = torch.cat([h_f, h_b], dim=-1)                 # [B, 2s]
        enriched_state = self.state_analyzer(final_state)           # [B, d]

        logits = self.classifier(pooled + enriched_state)           # [B, 3]
        return logits

class DirectionDatasetBERT(Dataset):
    def __init__(self, csv_path, max_length=64, use_bert_tokenizer=True, bert_model_name=DEFAULT_BERT_NAME):
        df = pd.read_csv(csv_path)
        required_cols = {'text', 'direction'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV 必须包含列: {required_cols}，实际: {df.columns.tolist()}")

        self.texts = df['text'].astype(str).tolist()
        raw_labels = df['direction'].tolist()

        label_map = {'left': 0, 'right': 1, 'bidirectional': 2}
        self.labels = []
        valid_indices = []
        for i, lbl in enumerate(raw_labels):
            lbl_low = str(lbl).strip().lower()
            if lbl_low in label_map:
                self.labels.append(label_map[lbl_low])
                valid_indices.append(i)

        self.texts = [self.texts[i] for i in valid_indices]
        if len(self.texts) == 0:
            raise ValueError("过滤后无有效样本。")

        print(f"✓ 有效样本数: {len(self.texts)} (原始 {len(df)} 条)")

        self.max_length = max_length
        self.use_bert_tokenizer = use_bert_tokenizer
        self.bert_model_name = bert_model_name

        if use_bert_tokenizer:
            if not HAS_TRANSFORMERS:
                raise RuntimeError("未安装 transformers，无法使用 BERT tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.vocab_size = self.tokenizer.vocab_size
            print(f"✓ 使用 BERT Tokenizer: {bert_model_name}, vocab_size={self.vocab_size}")
        else:
            all_chars = set(''.join(self.texts))
            self.char_to_idx = {ch: i+1 for i, ch in enumerate(sorted(all_chars))}
            self.char_to_idx['<PAD>'] = 0
            self.vocab_size = len(self.char_to_idx)
            print(f"✓ 使用字符级 Tokenizer, vocab_size={self.vocab_size}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.use_bert_tokenizer:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            input_ids = enc['input_ids'].squeeze(0)
        else:
            input_ids = [self.char_to_idx.get(ch, 0) for ch in text[:self.max_length]]
            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        return {'input_ids': input_ids, 'direction': torch.tensor(label, dtype=torch.long)}

def _per_class_metrics(cm):
    num_classes = cm.shape[0]
    P, R, F1 = [], [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        P.append(p)
        R.append(r)
        F1.append(f1)
    macro_p = sum(P) / num_classes
    macro_r = sum(R) / num_classes
    macro_f1 = sum(F1) / num_classes
    return P, R, F1, macro_p, macro_r, macro_f1

def train_mamba_de(model, train_loader, val_loader, num_epochs=30, lr=1e-3, device='cuda',
                   save_path='checkpoints/best_model.pth', use_focal=False, focal_gamma=2.0,
                   early_stop_patience=5, clip_max_norm=1.0, extra_meta=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.to(device)

    # 统计类权重
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['direction'].tolist())
    label_counts = torch.bincount(torch.tensor(all_labels))
    total = label_counts.sum()
    class_weights = total / (label_counts.float() * 3 + 1e-6)
    class_weights = class_weights.to(device)
    print(f"类别权重: {class_weights.tolist()}")

    # 损失函数
    if use_focal:
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
        print(f"使用 Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # AMP 控制：尊重设备与可选环境变量 SDM_NO_AMP（由 main 在 --no-amp 时设置）
    use_amp = (device == 'cuda') and (os.environ.get('SDM_NO_AMP', '0') != '1')
    scaler = _make_scaler(enabled=use_amp)

    best_val_acc = 0.0
    patience = early_stop_patience
    no_improve = 0
    direction_names = ['left', 'right', 'bidirectional']

    for epoch in range(num_epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['direction'].to(device)
            optimizer.zero_grad()

            with _autocast_ctx(enabled=use_amp):
                logits = model(input_ids)
                loss = criterion(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
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
            print(f'{header:12s}  ' + '  '.join([f'{n:12s}' for n in direction_names]))
            for i, row in enumerate(cm):
                print(f'{direction_names[i]:12s}  ' + '  '.join([f'{v:12d}' for v in row]))
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
                'd_state': model.d_state,
                'use_bert_tokenizer': ds_flag,
                'freeze_embedding': model.freeze_embedding,
                'bert_model_name': getattr(model, 'bert_model_name', DEFAULT_BERT_NAME),
                'pad_token_id': getattr(model, 'pad_token_id', 0),
                'class_weights': class_weights.detach().cpu().tolist(),
                'use_focal': use_focal,
                'focal_gamma': focal_gamma,
                'dropout': getattr(model, 'dropout', 0.5),
            }
            if char_to_idx is not None:
                save_dict['char_to_idx'] = char_to_idx
            if extra_meta:
                save_dict.update(extra_meta)

            torch.save(save_dict, save_path)
            print(f'  ✓ 保存最佳模型 -> {save_path} (验证准确率: {val_acc:.2f}%)\n')
        else:
            no_improve += 1
            print(f'  ↳ 验证未提升（连续 {no_improve}/{patience}）\n')
            if no_improve >= patience:
                print(f'⏹ 触发 EarlyStopping（patience={patience}），提前结束训练。')
                break

        scheduler.step()

    print(f'\n训练完成！最佳验证准确率: {best_val_acc:.2f}%')
    return model, device

def _build_cli():
    import argparse
    p = argparse.ArgumentParser(description="Mamba DE 训练（多语种 + 预训练Embedding + AMP + 早停）")
    p.add_argument('--csv', type=str, default='training_data.csv', help='训练CSV路径')
    p.add_argument('--bert-model', type=str, default=DEFAULT_BERT_NAME, help='预训练模型名/路径')
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--d-state', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--freeze-embedding', action='store_true', help='冻结embedding')
    p.add_argument('--max-length', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--train-split', type=float, default=0.8)
    p.add_argument('--use-focal', action='store_true')
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--early-stop', type=int, default=5, help='EarlyStopping patience')
    p.add_argument('--clip-max-norm', type=float, default=1.0)
    p.add_argument('--no-amp', action='store_true', help='禁用AMP')
    p.add_argument('--save-name', type=str, default='best_model_bert.pth', help='保存文件名')
    return p

def main():
    print("="*60)
    print("Mamba Direction Estimator - ALL BUGS FIXED")
    print("="*60)

    if not HAS_TRANSFORMERS:
        print("\n❌ 需要安装 transformers：pip install transformers")
        return

    args = _build_cli().parse_args()

    # 设备自动选择；不改 CLI 接口
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.no_amp:
        os.environ['SDM_NO_AMP'] = '1'  # train 内部读取，保持函数签名不变

    dataset = DirectionDatasetBERT(
        csv_path=args.csv,
        max_length=args.max_length,
        use_bert_tokenizer=True,
        bert_model_name=args.bert_model
    )

    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    # 固定划分种子，增强可复现性；不改函数/参数接口
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"✓ 训练集: {train_size} | 验证集: {val_size}")

    model = MambaDirectionEstimator(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        d_state=args.d_state,
        dropout=args.dropout,
        freeze_embedding=args.freeze_embedding,
        bert_model_name=args.bert_model
    )

    save_path = os.path.join('checkpoints', args.save_name)
    trained_model, device = train_mamba_de(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=dev,  # 不再写死 'cuda'
        save_path=save_path,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        early_stop_patience=args.early_stop,
        clip_max_norm=args.clip_max_norm,
        extra_meta={
            'max_length': args.max_length,
            'config': {
                'bert_model_name': args.bert_model,
                'max_length': args.max_length,
                'batch_size': args.batch_size,
                'num_epochs': args.epochs,
                'learning_rate': args.lr,
                'train_split': args.train_split,
                'd_model': args.d_model,
                'd_state': args.d_state,
                'dropout': args.dropout,
                'freeze_embedding': args.freeze_embedding
            }
        }
    )

    print(f"\n✅ 训练完成，模型已保存: {save_path}")

if __name__ == '__main__':
    main()
