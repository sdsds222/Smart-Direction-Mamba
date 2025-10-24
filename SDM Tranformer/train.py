# train.py (Transformer Version)
import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from typing import List
from torch.utils.data import DataLoader, Subset

from transformer_de import (
    DirectionDatasetBERT,
    TransformerDirectionEstimator,
    train_transformer_de,
)

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def list_csv_files(directory='.') -> List[str]:
    return sorted(glob.glob(os.path.join(directory, '*.csv')))


def select_multiple_csv(csv_files: List[str]) -> List[str]:
    print("\n" + "="*70)
    print("📚 多数据集选择")
    print("="*70)
    print("\n提示：输入数字序号，用逗号或空格分隔（如: 1,2,3 或 1 2 3）")
    print("输入 'all' 选择全部\n")

    for i, file in enumerate(csv_files, 1):
        size_kb = os.path.getsize(file) / 1024
        try:
            total_rows = sum(1 for _ in open(file, encoding='utf-8')) - 1
            print(f"  {i}. {os.path.basename(file):30s} ({size_kb:.1f} KB, ~{total_rows} 行)")
        except Exception:
            print(f"  {i}. {os.path.basename(file):30s} ({size_kb:.1f} KB)")

    while True:
        selection = input(f"\n请选择要合并的数据集: ").strip()
        if selection.lower() == 'all':
            return csv_files

        try:
            if ',' in selection:
                indices = [int(x.strip()) for x in selection.split(',') if x.strip()]
            else:
                indices = [int(x.strip()) for x in selection.split() if x.strip()]
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
                print("❌ 未选择任何文件")
        except ValueError:
            print("❌ 输入格式错误，请输入数字序号")


def select_csv_interactive() -> List[str]:
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
        size_kb = os.path.getsize(file) / 1024
        try:
            total_rows = sum(1 for _ in open(file, encoding='utf-8')) - 1
            print(f"  {i}. {os.path.basename(file):30s} ({size_kb:.1f} KB, ~{total_rows} 行)")
        except Exception:
            print(f"  {i}. {os.path.basename(file):30s} ({size_kb:.1f} KB)")

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
            print(f"❌ 请输入 1 - {len(csv_files)+2} 之间的数字")


def merge_datasets(csv_paths: List[str]) -> str:
    print("\n" + "="*70)
    print("🔄 合并数据集")
    print("="*70)

    all_dfs = []
    total_samples = 0

    for csv_path in csv_paths:
        print(f"\n读取: {os.path.basename(csv_path)}")
        df = pd.read_csv(csv_path, encoding='utf-8')
        if 'text' not in df.columns or 'direction' not in df.columns:
            print(f"  ⚠️ 跳过: 缺少必需列 (text, direction)")
            continue
        print(f"  ✓ {len(df)} 条样本")
        all_dfs.append(df)
        total_samples += len(df)

    if not all_dfs:
        raise ValueError("没有有效的数据集可以合并")

    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✓ 合并完成: 总计 {total_samples} 条样本")

    merged_path = 'merged_dataset.csv'
    merged_df.to_csv(merged_path, index=False, encoding='utf-8')
    print(f"✓ 合并数据集已保存至: {merged_path}")
    return merged_path


def preview_dataset(csv_paths: List[str]) -> bool:
    print("\n" + "="*70)
    print("📊 数据集预览")
    print("="*70)

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
                print(f"   方向分布: " + ', '.join([f"{d}:{c}" for d, c in direction_counts.items()]))
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

            max_ratio = direction_counts.max() / max(1, direction_counts.min())
            if max_ratio > 3:
                print(f"  ⚠️ 数据不平衡（最大/最小 = {max_ratio:.2f}）")

        if total_samples < 50:
            print(f"  ⚠️ 样本数量较少（建议至少100条）")
        return True

    csv_path = csv_paths[0]
    df = pd.read_csv(csv_path, encoding='utf-8')

    print(f"\n文件: {os.path.basename(csv_path)}")
    print(f"总样本数: {len(df)}")
    print(f"列名: {list(df.columns)}")

    if 'text' not in df.columns:
        print("\n⚠️ 警告: 找不到 'text' 列")
        return False
    if 'direction' not in df.columns:
        print("\n⚠️ 警告: 找不到 'direction' 列")
        return False

    print("\n方向分布:")
    direction_counts = df['direction'].value_counts()
    for direction, count in direction_counts.items():
        pct = 100 * count / len(df)
        print(f"  {direction}: {count} ({pct:.1f}%)")

    max_ratio = direction_counts.max() / max(1, direction_counts.min())
    if max_ratio > 3:
        print(f"\n⚠️ 数据不平衡（最大/最小 = {max_ratio:.2f}）")
    if len(df) < 50:
        print(f"\n⚠️ 样本数量较少（建议至少100条）")

    print("\n前3条样本:")
    print("-"*70)
    for i, row in df.head(3).iterrows():
        text = str(row['text'])[:50]
        direction = row['direction']
        print(f"{i+1}. [{direction}] {text}...")
    return True


def _labels_from_dataset(ds: DirectionDatasetBERT) -> np.ndarray:
    return np.array(ds.directions, dtype=np.int64)


def stratified_split_indices(labels: np.ndarray, train_ratio=0.8, seed=42):
    rs = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    num_classes = int(labels.max()) + 1
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rs.shuffle(idx_c)
        n_train = int(len(idx_c) * train_ratio)
        train_idx.extend(idx_c[:n_train].tolist())
        val_idx.extend(idx_c[n_train:].tolist())
    rs.shuffle(train_idx)
    rs.shuffle(val_idx)
    return train_idx, val_idx


def train_model(csv_path: str, config: dict):
    print("\n" + "="*70)
    print("🚀 开始训练（Transformer + 分层划分 + AMP + EarlyStopping）")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ 使用设备: {device}")

    print("\n[1/4] 加载数据集...")
    dataset = DirectionDatasetBERT(
        csv_path=csv_path,
        max_length=config['max_length'],
        use_bert_tokenizer=True,
        bert_model_name=config['bert_model']
    )

    labels = _labels_from_dataset(dataset)
    train_idx, val_idx = stratified_split_indices(labels, train_ratio=config['train_split'], seed=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False)

    print(f"✓ 训练集: {len(train_dataset)} 样本")
    print(f"✓ 验证集: {len(val_dataset)} 样本（分层切分）")

    print("\n[2/4] 初始化模型...")
    model = TransformerDirectionEstimator(
        vocab_size=dataset.vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config.get('dropout', 0.5),
        freeze_embedding=config.get('freeze_embedding', False),
        bert_model_name=config['bert_model']
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 模型总参数: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"✓ 可训练参数: {num_trainable:,} ({num_trainable/1e6:.2f}M)")

    print("\n[3/4] 开始训练（调用 transformer_de.train_transformer_de）...")
    os.makedirs('checkpoints', exist_ok=True)
    save_path = os.path.join('checkpoints', f"{config['model_name']}.pth")

    trained_model, device = train_transformer_de(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_path=save_path,
        use_focal=config['use_focal'],
        focal_gamma=config['focal_gamma'],
        early_stop_patience=config['early_stop'],
        clip_max_norm=config['clip_max_norm'],
        extra_meta={
            'max_length': config['max_length'],
            'config': config
        }
    )

    print(f"\n[4/4] 最佳模型已保存至: {save_path}")
    print("="*70)
    print("✅ 训练完成（详见日志：Per-class 指标 / 混淆矩阵 / EarlyStopping 等）")
    print("="*70)


def build_cli():
    p = argparse.ArgumentParser(
        description='交互式训练 Transformer Direction Estimator (多数据集 + 预训练Embedding + 分层划分)'
    )
    p.add_argument('-i', '--input', type=str, nargs='+', help='训练数据 CSV 文件路径（可多个）')
    p.add_argument('--auto', action='store_true', help='使用默认配置，不交互')
    p.add_argument('--bert-model', type=str, default='xlm-roberta-base', help='预训练模型名/路径')
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--nhead', type=int, default=4, help='注意力头数')
    p.add_argument('--num-layers', type=int, default=2, help='Transformer层数')
    p.add_argument('--dim-feedforward', type=int, default=512, help='FFN隐藏层维度')
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--freeze-embedding', action='store_true', help='冻结预训练 embedding（默认微调）')
    p.add_argument('--max-length', type=int, default=64)
    p.add_argument('--train-split', type=float, default=0.8, help='训练集比例（分层切分）')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--use-focal', action='store_true', help='使用 FocalLoss（默认关闭）')
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--early-stop', type=int, default=5, help='EarlyStopping patience')
    p.add_argument('--clip-max-norm', type=float, default=1.0)
    p.add_argument('--model-name', type=str, default='best_model_transformer', help='保存名称（会保存到 checkpoints/{name}.pth）')
    return p


def main():
    args = build_cli().parse_args()

    print("="*70)
    print("🎓 Transformer Direction Estimator - 交互式训练 (多数据集 + 预训练Embedding + 分层划分)")
    print("="*70)

    if not HAS_TRANSFORMERS:
        print("\n⚠️ 未安装 transformers，将无法使用预训练分词器/Embedding：pip install transformers")
        cont = input("是否继续（将退回字符级Tokenizer，不建议）？[y/N]: ").strip().lower()
        if cont != 'y':
            print("训练取消")
            return

    if args.input:
        csv_paths = args.input
        for pth in csv_paths:
            if not os.path.exists(pth):
                print(f"❌ 错误: 找不到文件 {pth}")
                return
    else:
        csv_paths = select_csv_interactive()

    if not preview_dataset(csv_paths):
        print("\n❌ 数据集格式错误，训练取消")
        return

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

    config = {
        'bert_model': args.bert_model,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'train_split': args.train_split,
        'dropout': args.dropout,
        'model_name': args.model_name,
        'freeze_embedding': args.freeze_embedding,
        'use_focal': args.use_focal,
        'focal_gamma': args.focal_gamma,
        'early_stop': args.early_stop,
        'clip_max_norm': args.clip_max_norm,
    }

    print("\n最终配置:")
    print("-"*70)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("-"*70)

    if not args.auto:
        go = input("\n开始训练? [Y/n]: ").strip().lower()
        if go == 'n':
            print("训练取消")
            return

    try:
        train_model(csv_path, config)
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()