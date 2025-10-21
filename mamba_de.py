"""
Mamba DE - 使用真实的BERT Tokenizer
这个版本使用的tokenization方法和BERT一样，更接近实际大模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os

# 需要安装: pip install transformers
try:
    from transformers import BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  警告: 未安装transformers库")
    print("运行: pip install transformers")


class SimplifiedMambaSSM(nn.Module):
    """简化的Mamba SSM单元"""
    
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.input_gate = nn.Linear(d_model, d_model)
        
    def forward(self, x, h=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if h is None:
            h = torch.zeros(batch_size, self.d_state, device=device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            x_gated = torch.sigmoid(self.input_gate(x_t)) * x_t
            
            B_t = self.B_proj(x_gated)
            C_t = self.C_proj(x_gated)
            
            A_mean = self.A.mean(dim=0)
            h = torch.tanh(h * A_mean + B_t)
            
            y_t = (C_t * h).sum(dim=-1, keepdim=True) * torch.ones_like(x_t) + self.D * x_t
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        return output, h


class MambaDirectionEstimator(nn.Module):
    """Mamba方向判别器 - 使用BERT tokenizer"""
    
    def __init__(self, vocab_size: int, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 嵌入层（和BERT一样的做法）
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        self.forward_ssm = SimplifiedMambaSSM(d_model, d_state)
        self.backward_ssm = SimplifiedMambaSSM(d_model, d_state)
        
        self.state_analyzer = nn.Sequential(
            nn.Linear(d_state * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.decision_head = nn.Linear(d_model // 2, 3)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        _, h_forward = self.forward_ssm(x)
        x_reversed = torch.flip(x, dims=[1])
        _, h_backward = self.backward_ssm(x_reversed)
        
        h_combined = torch.cat([h_forward, h_backward], dim=-1)
        features = self.state_analyzer(h_combined)
        direction_logits = self.decision_head(features)
        
        return direction_logits
    
    def predict(self, input_ids):
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            probs = F.softmax(logits, dim=-1)
            directions = torch.argmax(probs, dim=-1)
        return directions, probs


class DirectionDatasetBERT(Dataset):
    """使用BERT Tokenizer的数据集"""
    
    def __init__(self, csv_path: str, max_length: int = 64, use_bert_tokenizer: bool = True):
        df = pd.read_csv(csv_path, encoding='utf-8')
        self.texts = df['text'].tolist()
        self.directions = df['direction'].tolist()
        
        self.max_length = max_length
        self.use_bert_tokenizer = use_bert_tokenizer and HAS_TRANSFORMERS
        
        self.direction_map = {
            'left': 0, 'right': 1, 'bidirectional': 2,
            '左': 0, '右': 1, '双向': 2,
            'L': 0, 'R': 1, 'B': 2
        }
        
        # 初始化tokenizer
        if self.use_bert_tokenizer:
            print("✓ 使用BERT Tokenizer (和真实大模型一样的方法)")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.vocab_size = self.tokenizer.vocab_size
        else:
            print("✓ 使用简单字符级Tokenizer")
            self._build_char_vocab()
        
        print(f"✓ 加载数据集: {len(self.texts)} 条样本")
        print(f"✓ 词表大小: {self.vocab_size}")
        self._print_statistics()
    
    def _build_char_vocab(self):
        """后备方案：字符级tokenizer"""
        chars = set()
        for text in self.texts:
            chars.update(text)
        
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, char in enumerate(sorted(chars)):
            self.char_to_idx[char] = idx + 2
        
        self.vocab_size = len(self.char_to_idx)
    
    def _print_statistics(self):
        direction_counts = {}
        for d in self.directions:
            key = str(d)
            direction_counts[key] = direction_counts.get(key, 0) + 1
        
        print("✓ 方向分布:")
        for direction, count in sorted(direction_counts.items()):
            pct = 100 * count / len(self.directions)
            print(f"  {direction}: {count} ({pct:.1f}%)")
    
    def tokenize(self, text: str):
        """Tokenization"""
        if self.use_bert_tokenizer:
            # 🚀 使用BERT的子词级tokenization
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            return torch.tensor(encoded, dtype=torch.long)
        else:
            # 后备：字符级tokenization
            tokens = [self.char_to_idx.get(char, 1) for char in text[:self.max_length]]
            if len(tokens) < self.max_length:
                tokens = tokens + [0] * (self.max_length - len(tokens))
            return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        direction = self.directions[idx]
        
        input_ids = self.tokenize(text)
        direction_label = self.direction_map.get(direction, 0)
        
        return {
            'input_ids': input_ids,
            'direction': torch.tensor(direction_label, dtype=torch.long),
            'text': text
        }


def train_mamba_de(model, train_loader, val_loader, num_epochs=20, lr=1e-3, device='cuda'):
    """训练函数"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ 使用设备: {device}\n")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    direction_names = ['左向(因果)', '右向(反因果)', '双向']
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['direction'].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        confusion_matrix = np.zeros((3, 3), dtype=int)
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['direction'].to(device)
                
                logits = model(input_ids)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    confusion_matrix[true][pred] += 1
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        if (epoch + 1) % 5 == 0:
            print('\n混淆矩阵:')
            header = '实际\\预测'
            print(f'{header:12s}  ' + '  '.join([f'{name:12s}' for name in direction_names]))
            for i, row in enumerate(confusion_matrix):
                print(f'{direction_names[i]:12s}  ' + '  '.join([f'{val:12d}' for val in row]))
        
        print()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # 保存tokenizer类型
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'vocab_size': model.embedding.num_embeddings,
                'd_model': model.d_model,
                'd_state': model.d_state,
                'use_bert_tokenizer': train_loader.dataset.dataset.use_bert_tokenizer
            }
            
            # 如果用的是字符级，保存词表
            if not train_loader.dataset.dataset.use_bert_tokenizer:
                save_dict['char_to_idx'] = train_loader.dataset.dataset.char_to_idx
            
            torch.save(save_dict, 'checkpoints/best_model_bert.pth')
            print(f'✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)\n')
        
        scheduler.step()
    
    print(f'\n训练完成！最佳验证准确率: {best_val_acc:.2f}%')
    return model, device


def main():
    """主函数"""
    print("="*60)
    print("Mamba Direction Estimator - BERT Tokenizer版")
    print("="*60)
    
    if not HAS_TRANSFORMERS:
        print("\n❌ 错误: 需要安装transformers库")
        print("运行: pip install transformers")
        print("安装后会自动下载BERT中文模型（约400MB）")
        return
    
    D_MODEL = 128
    D_STATE = 16
    MAX_LENGTH = 64
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    TRAIN_SPLIT = 0.8
    
    print("\n[1/5] 加载数据集...")
    dataset = DirectionDatasetBERT(
        csv_path='training_data.csv',
        max_length=MAX_LENGTH,
        use_bert_tokenizer=True  # 使用BERT tokenizer
    )
    
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ 训练集: {train_size} 样本")
    print(f"✓ 验证集: {val_size} 样本")
    
    print("\n[2/5] 初始化模型...")
    model = MambaDirectionEstimator(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL,
        d_state=D_STATE,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    print("\n[3/5] 开始训练...")
    print("💡 提示: 使用BERT tokenizer后，模型性能应该会更好")
    
    trained_model, device = train_mamba_de(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device='cuda'
    )
    
    print("\n[4/5] 测试推理...")
    trained_model.eval()
    sample_batch = next(iter(val_loader))
    
    direction_names = ['左向(因果)', '右向(反因果)', '双向']
    
    with torch.no_grad():
        input_ids = sample_batch['input_ids'][:5].to(device)
        texts = sample_batch['text'][:5]
        true_labels = sample_batch['direction'][:5]
        
        directions, probs = trained_model.predict(input_ids)
        
        directions = directions.cpu()
        probs = probs.cpu()
        
        print("\n推理示例:")
        print("-" * 60)
        for i in range(len(texts)):
            print(f"\n文本: {texts[i]}")
            print(f"真实标签: {direction_names[true_labels[i]]}")
            print(f"预测方向: {direction_names[directions[i]]}")
            print(f"置信度: ", end='')
            for j, prob in enumerate(probs[i]):
                print(f"{direction_names[j]}: {prob:.3f}  ", end='')
            print()
    
    print("\n[5/5] 完成！")
    print(f"✓ 模型已保存至: checkpoints/best_model_bert.pth")
    print("\n💡 这个模型使用的tokenization方法和BERT一样！")
    print("="*60)


if __name__ == '__main__':
    main()