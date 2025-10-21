"""
交互式训练脚本 - 支持多数据集合并训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import argparse
import glob

try:
    from transformers import BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from mamba_de import SimplifiedMambaSSM, MambaDirectionEstimator


def list_csv_files(directory='.'):
    """列出当前目录下的所有CSV文件"""
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    return sorted(csv_files)


def select_multiple_csv(csv_files):
    """选择多个CSV文件"""
    print("\n" + "="*70)
    print("📚 多数据集选择")
    print("="*70)
    print("\n提示：输入数字序号，用逗号或空格分隔")
    print("例如: 1,2,3 或 1 2 3")
    print("输入 'all' 选择全部\n")
    
    for i, file in enumerate(csv_files, 1):
        size = os.path.getsize(file) / 1024
        try:
            total_rows = sum(1 for _ in open(file, encoding='utf-8')) - 1
            print(f"  {i}. {os.path.basename(file):30s} ({size:.1f} KB, ~{total_rows} 行)")
        except:
            print(f"  {i}. {os.path.basename(file):30s} ({size:.1f} KB)")
    
    while True:
        selection = input(f"\n请选择要合并的数据集: ").strip()
        
        if selection.lower() == 'all':
            return csv_files
        
        # 解析输入
        try:
            # 支持逗号或空格分隔
            if ',' in selection:
                indices = [int(x.strip()) for x in selection.split(',')]
            else:
                indices = [int(x.strip()) for x in selection.split()]
            
            # 验证索引
            selected_files = []
            for idx in indices:
                if 1 <= idx <= len(csv_files):
                    selected_files.append(csv_files[idx-1])
                else:
                    print(f"❌ 序号 {idx} 无效")
                    break
            else:
                if selected_files:
                    print(f"\n✓ 已选择 {len(selected_files)} 个数据集:")
                    for f in selected_files:
                        print(f"  - {os.path.basename(f)}")
                    return selected_files
                else:
                    print("❌ 未选择任何文件")
        except ValueError:
            print("❌ 输入格式错误，请输入数字序号")


def select_csv_interactive():
    """交互式选择CSV文件（支持多选）"""
    print("\n" + "="*70)
    print("📁 选择训练数据集")
    print("="*70)
    
    csv_files = list_csv_files()
    
    if not csv_files:
        print("❌ 当前目录没有找到CSV文件")
        csv_path = input("\n请输入CSV文件路径: ").strip()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到文件: {csv_path}")
        return [csv_path]
    
    print("\n找到以下CSV文件:")
    for i, file in enumerate(csv_files, 1):
        size = os.path.getsize(file) / 1024
        try:
            total_rows = sum(1 for _ in open(file, encoding='utf-8')) - 1
            print(f"  {i}. {os.path.basename(file):30s} ({size:.1f} KB, ~{total_rows} 行)")
        except:
            print(f"  {i}. {os.path.basename(file):30s} ({size:.1f} KB)")
    
    print(f"  {len(csv_files)+1}. 手动输入文件路径")
    print(f"  {len(csv_files)+2}. 选择多个数据集")
    
    while True:
        choice = input(f"\n请选择 [1-{len(csv_files)+2}]: ").strip()
        
        if not choice.isdigit():
            print("❌ 请输入数字")
            continue
        
        choice = int(choice)
        
        if 1 <= choice <= len(csv_files):
            return [csv_files[choice-1]]
        elif choice == len(csv_files) + 1:
            csv_path = input("请输入CSV文件路径: ").strip()
            if not os.path.exists(csv_path):
                print(f"❌ 找不到文件: {csv_path}")
                continue
            return [csv_path]
        elif choice == len(csv_files) + 2:
            return select_multiple_csv(csv_files)
        else:
            print(f"❌ 请输入1-{len(csv_files)+2}之间的数字")


def merge_datasets(csv_paths):
    """合并多个CSV文件"""
    print("\n" + "="*70)
    print("🔄 合并数据集")
    print("="*70)
    
    all_dfs = []
    total_samples = 0
    
    for csv_path in csv_paths:
        print(f"\n读取: {os.path.basename(csv_path)}")
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # 检查必需列
        if 'text' not in df.columns or 'direction' not in df.columns:
            print(f"  ⚠️  跳过: 缺少必需列 (需要 text 和 direction)")
            continue
        
        print(f"  ✓ {len(df)} 条样本")
        all_dfs.append(df)
        total_samples += len(df)
    
    if not all_dfs:
        raise ValueError("没有有效的数据集可以合并")
    
    # 合并
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n✓ 合并完成: 总计 {total_samples} 条样本")
    
    # 保存合并后的数据集
    merged_path = 'merged_dataset.csv'
    merged_df.to_csv(merged_path, index=False, encoding='utf-8')
    print(f"✓ 合并数据集已保存至: {merged_path}")
    
    return merged_path


def preview_dataset(csv_paths):
    """预览数据集（支持单个或多个）"""
    print("\n" + "="*70)
    print("📊 数据集预览")
    print("="*70)
    
    # 如果是多个文件，先显示各自的信息
    if len(csv_paths) > 1:
        print(f"\n共选择 {len(csv_paths)} 个数据集:")
        
        total_samples = 0
        all_directions = []
        
        for i, csv_path in enumerate(csv_paths, 1):
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"\n{i}. {os.path.basename(csv_path)}")
            print(f"   样本数: {len(df)}")
            
            if 'direction' in df.columns:
                direction_counts = df['direction'].value_counts()
                print(f"   方向分布: ", end='')
                print(', '.join([f"{d}:{c}" for d, c in direction_counts.items()]))
                all_directions.extend(df['direction'].tolist())
            
            total_samples += len(df)
        
        print(f"\n{'='*70}")
        print(f"合并后统计:")
        print(f"  总样本数: {total_samples}")
        
        if all_directions:
            direction_counts = pd.Series(all_directions).value_counts()
            print(f"  方向分布:")
            for direction, count in direction_counts.items():
                pct = 100 * count / len(all_directions)
                print(f"    {direction}: {count} ({pct:.1f}%)")
            
            # 检查平衡性
            max_ratio = direction_counts.max() / direction_counts.min()
            if max_ratio > 3:
                print(f"  ⚠️  数据不平衡（最大/最小 = {max_ratio:.2f}）")
        
        if total_samples < 50:
            print(f"  ⚠️  样本数量较少（建议至少100条）")
        
        return True
    
    # 单个文件的预览
    csv_path = csv_paths[0]
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    print(f"\n文件: {os.path.basename(csv_path)}")
    print(f"总样本数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    
    if 'text' not in df.columns:
        print("\n⚠️  警告: 找不到 'text' 列")
        return False
    
    if 'direction' not in df.columns:
        print("\n⚠️  警告: 找不到 'direction' 列")
        return False
    
    print("\n方向分布:")
    direction_counts = df['direction'].value_counts()
    for direction, count in direction_counts.items():
        pct = 100 * count / len(df)
        print(f"  {direction}: {count} ({pct:.1f}%)")
    
    max_ratio = direction_counts.max() / direction_counts.min()
    if max_ratio > 3:
        print(f"\n⚠️  警告: 数据不平衡（最大/最小 = {max_ratio:.2f}）")
    
    if len(df) < 50:
        print(f"\n⚠️  警告: 样本数量较少（建议至少100条）")
    
    print("\n前3条样本:")
    print("-"*70)
    for i, row in df.head(3).iterrows():
        text = str(row['text'])[:50]
        direction = row['direction']
        print(f"{i+1}. [{direction}] {text}...")
    
    return True


def get_training_config():
    """交互式获取训练配置"""
    print("\n" + "="*70)
    print("⚙️  训练配置")
    print("="*70)
    
    configs = {}
    
    print("\n1. 模型参数")
    configs['d_model'] = int(input("  嵌入维度 [默认: 128]: ").strip() or "128")
    configs['d_state'] = int(input("  状态维度 [默认: 16]: ").strip() or "16")
    configs['max_length'] = int(input("  最大序列长度 [默认: 64]: ").strip() or "64")
    
    print("\n2. 训练参数")
    configs['batch_size'] = int(input("  批次大小 [默认: 16]: ").strip() or "16")
    configs['num_epochs'] = int(input("  训练轮数 [默认: 30]: ").strip() or "30")
    configs['learning_rate'] = float(input("  学习率 [默认: 0.001]: ").strip() or "0.001")
    configs['train_split'] = float(input("  训练集比例 [默认: 0.8]: ").strip() or "0.8")
    
    print("\n3. 输出设置")
    configs['model_name'] = input("  模型保存名称 [默认: best_model]: ").strip() or "best_model"
    
    return configs


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
        
        if self.use_bert_tokenizer:
            print("✓ 使用BERT Tokenizer")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.vocab_size = self.tokenizer.vocab_size
        else:
            print("✓ 使用字符级Tokenizer")
            self._build_char_vocab()
        
        print(f"✓ 词表大小: {self.vocab_size}")
    
    def _build_char_vocab(self):
        chars = set()
        for text in self.texts:
            chars.update(text)
        
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, char in enumerate(sorted(chars)):
            self.char_to_idx[char] = idx + 2
        
        self.vocab_size = len(self.char_to_idx)
    
    def tokenize(self, text: str):
        if self.use_bert_tokenizer:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            return torch.tensor(encoded, dtype=torch.long)
        else:
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


def train_model(csv_path, config):
    """训练模型"""
    print("\n" + "="*70)
    print("🚀 开始训练")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ 使用设备: {device}")
    
    print("\n[1/4] 加载数据集...")
    dataset = DirectionDatasetBERT(
        csv_path=csv_path,
        max_length=config['max_length'],
        use_bert_tokenizer=True
    )
    
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"✓ 训练集: {train_size} 样本")
    print(f"✓ 验证集: {val_size} 样本")
    
    print("\n[2/4] 初始化模型...")
    model = MambaDirectionEstimator(
        vocab_size=dataset.vocab_size,
        d_model=config['d_model'],
        d_state=config['d_state'],
        dropout=0.1
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n[3/4] 开始训练...")
    print("-"*70)
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
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
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
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
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}] '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            model_path = f'checkpoints/{config["model_name"]}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'vocab_size': model.embedding.num_embeddings,
                'd_model': model.d_model,
                'd_state': model.d_state,
                'use_bert_tokenizer': dataset.use_bert_tokenizer,
                'config': config
            }, model_path)
            
            print(f'  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)')
        
        scheduler.step()
    
    print("\n[4/4] 训练完成!")
    print("="*70)
    print(f"✅ 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"✅ 模型已保存至: checkpoints/{config['model_name']}.pth")
    print("="*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='交互式训练 Mamba Direction Estimator (支持多数据集)')
    parser.add_argument('-i', '--input', type=str, nargs='+', help='训练数据CSV文件路径（可以多个）')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--model-name', type=str, help='模型保存名称')
    parser.add_argument('--auto', action='store_true', help='使用默认配置，不交互')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎓 Mamba Direction Estimator - 交互式训练 (多数据集支持)")
    print("="*70)
    
    if not HAS_TRANSFORMERS:
        print("\n⚠️  警告: 未安装transformers库")
        print("建议安装: pip install transformers")
        use_bert = input("\n是否继续使用字符级tokenizer? [y/N]: ").strip().lower()
        if use_bert != 'y':
            print("训练取消")
            return
    
    # 1. 选择CSV文件
    if args.input:
        csv_paths = args.input
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"❌ 错误: 找不到文件 {csv_path}")
                return
    else:
        csv_paths = select_csv_interactive()
    
    # 2. 预览数据集
    if not preview_dataset(csv_paths):
        print("\n❌ 数据集格式错误，训练取消")
        return
    
    # 3. 合并数据集（如果有多个）
    if len(csv_paths) > 1:
        confirm = input("\n是否合并这些数据集进行训练? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print("训练取消")
            return
        
        csv_path = merge_datasets(csv_paths)
    else:
        csv_path = csv_paths[0]
        confirm = input("\n是否使用此数据集训练? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print("训练取消")
            return
    
    # 4. 获取训练配置
    if args.auto:
        print("\n使用默认配置...")
        config = {
            'd_model': 128,
            'd_state': 16,
            'max_length': 64,
            'batch_size': args.batch_size or 16,
            'num_epochs': args.epochs or 30,
            'learning_rate': args.lr or 0.001,
            'train_split': 0.8,
            'model_name': args.model_name or 'best_model'
        }
    else:
        config = get_training_config()
        
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.epochs:
            config['num_epochs'] = args.epochs
        if args.lr:
            config['learning_rate'] = args.lr
        if args.model_name:
            config['model_name'] = args.model_name
    
    print("\n最终配置:")
    print("-"*70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-"*70)
    
    if not args.auto:
        confirm = input("\n开始训练? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print("训练取消")
            return
    
    # 5. 开始训练
    try:
        train_model(csv_path, config)
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()