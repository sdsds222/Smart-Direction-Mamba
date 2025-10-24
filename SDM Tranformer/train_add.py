# train_add.py (Transformer Version)
import os
import argparse
import numpy as np
import torch
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


def fine_tune(base_model_path: str, data_csv: str, cfg: dict):
    print("\n" + "="*70)
    print("🔄 增量训练 (Fine-tuning) - Transformer / 分层划分 / AMP / EarlyStopping")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ 使用设备: {device}")

    print(f"\n[1/5] 加载基础模型: {os.path.basename(base_model_path)}")
    base_ckpt = torch.load(base_model_path, map_location='cpu')

    bert_in_ckpt = base_ckpt.get('bert_model_name')
    if not bert_in_ckpt:
        cfg_in_ckpt = base_ckpt.get('config', {}) if isinstance(base_ckpt.get('config', {}), dict) else {}
        for k in ['bert_model_name', 'bert_name', 'pretrained_name', 'tokenizer_name', 'hf_model_name']:
            if k in cfg_in_ckpt:
                bert_in_ckpt = cfg_in_ckpt[k]
                break
    bert_model = cfg.get('bert_model') or bert_in_ckpt or 'xlm-roberta-base'
    if cfg.get('bert_model') and cfg['bert_model'] != bert_in_ckpt:
        print(f"ℹ️ 使用命令行覆盖预训练模型: {bert_in_ckpt} → {cfg['bert_model']}")
    print(f"✓ 预训练模型: {bert_model}")

    freeze_from_ckpt = bool(base_ckpt.get('freeze_embedding', False))
    freeze_embedding = cfg.get('freeze_embedding')
    if freeze_embedding is None:
        freeze_embedding = freeze_from_ckpt
    else:
        if freeze_embedding != freeze_from_ckpt:
            print(f"ℹ️ 冻结策略覆盖: {freeze_from_ckpt} → {freeze_embedding}")
    print(f"✓ 冻结 embedding: {freeze_embedding}")

    dropout = cfg.get('dropout', None)
    if dropout is None:
        dropout = base_ckpt.get('dropout', base_ckpt.get('config', {}).get('dropout', 0.5))

    pad_token_id = base_ckpt.get('pad_token_id', 0)

    # Transformer参数
    nhead = base_ckpt.get('nhead', base_ckpt.get('config', {}).get('nhead', 4))
    num_layers = base_ckpt.get('num_layers', base_ckpt.get('config', {}).get('num_layers', 2))
    dim_feedforward = base_ckpt.get('dim_feedforward', base_ckpt.get('config', {}).get('dim_feedforward', 512))
    
    # 显式Attention分析参数（兼容旧模型）
    use_explicit_attention = base_ckpt.get('use_explicit_attention', base_ckpt.get('config', {}).get('use_explicit_attention', True))
    explicit_weight = base_ckpt.get('explicit_weight', base_ckpt.get('config', {}).get('explicit_weight', 0.3))

    model = TransformerDirectionEstimator(
        vocab_size=base_ckpt.get('vocab_size', 0),
        d_model=base_ckpt['d_model'],
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        freeze_embedding=freeze_embedding,
        bert_model_name=bert_model,
        use_explicit_attention=use_explicit_attention,
        explicit_weight=explicit_weight
    )
    model.load_state_dict(base_ckpt['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print(f"\n[2/5] 加载新数据并分层划分...")
    ds = DirectionDatasetBERT(
        csv_path=data_csv,
        max_length=cfg['max_length'],
        use_bert_tokenizer=True,
        bert_model_name=bert_model,
    )
    
    try:
        vocab_model = getattr(model.embedding, 'num_embeddings', None)
        vocab_tok = getattr(ds, 'vocab_size', None)
        if isinstance(vocab_model, int) and isinstance(vocab_tok, int) and vocab_model != vocab_tok:
            print(f"⚠️ 分词器词表({vocab_tok})与模型embedding词表({vocab_model})不一致。")
            print(f"   请确认 tokenizer 与基础模型一致（--bert-model）。")
    except Exception:
        pass

    labels = _labels_from_dataset(ds)
    if len(ds) < 10:
        print(f"⚠️ 新数据量过少 ({len(ds)} 条)，将使用全量训练，不划分验证集。")
        train_idx = np.arange(len(ds)).tolist()
        val_idx = []
    else:
        train_idx, val_idx = stratified_split_indices(labels, train_ratio=cfg['train_split'], seed=cfg['seed'])

    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx) if len(val_idx) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False) if val_ds is not None else None

    print(f"✓ 新训练集: {len(train_ds)} 样本")
    print(f"✓ 新验证集: {len(val_ds) if val_ds is not None else 0} 样本（分层切分）")

    base_lr = cfg['learning_rate']
    lr = base_lr * cfg['ft_scale']
    if cfg['ft_scale'] != 1.0:
        print(f"\n[3/5] 微调学习率缩放: {base_lr} × {cfg['ft_scale']} → {lr}")
    else:
        print(f"\n[3/5] 学习率: {lr}")

    print(f"\n[4/5] 开始增量训练（复用统一训练管线）...")
    os.makedirs('checkpoints', exist_ok=True)
    save_path = os.path.join('checkpoints', f"{cfg['model_name']}.pth")

    extra_meta = {
        'base_model': base_model_path,
        'base_val_acc': float(base_ckpt.get('val_acc', 0.0)),
        'fine_tuned': True,
        'max_length': cfg['max_length'],
        'config': cfg,
        'bert_model_name': bert_model,
        'pad_token_id': pad_token_id,
    }

    trained_model, _ = train_transformer_de(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if val_loader is not None else train_loader,
        num_epochs=cfg['num_epochs'],
        lr=lr,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_path=save_path,
        use_focal=cfg['use_focal'],
        focal_gamma=cfg['focal_gamma'],
        early_stop_patience=cfg['early_stop'],
        clip_max_norm=cfg['clip_max_norm'],
        extra_meta=extra_meta
    )

    print(f"\n[5/5] 增量训练完成！最佳模型已保存至: {save_path}")
    print("="*70)
    print("提示：预测时直接使用该路径，例如：")
    print(f"  python predict.py -m {save_path} -i your_eval.csv --batch-size 64")
    print("="*70)


def build_cli():
    p = argparse.ArgumentParser(description='增量训练 - 在已有模型基础上继续训练（Transformer版本）')
    p.add_argument('-m', '--model', type=str, required=True, help='基础模型路径（.pth）')
    p.add_argument('-i', '--input', type=str, required=True, help='新训练数据 CSV')
    p.add_argument('--bert-model', type=str, default=None, help='覆盖基础模型的预训练名称/路径（默认沿用 ckpt 里记录的）')
    p.add_argument('--freeze-embedding', dest='freeze_embedding', action='store_true', help='强制冻结预训练 embedding（默认沿用 ckpt）')
    p.add_argument('--no-freeze-embedding', dest='freeze_embedding', action='store_false', help='强制微调预训练 embedding（默认沿用 ckpt）')
    p.set_defaults(freeze_embedding=None)
    p.add_argument('--max-length', type=int, default=64, help='序列最大长度')
    p.add_argument('--train-split', type=float, default=0.8, help='训练集比例（分层切分）')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-3, help='基础学习率（将乘以 --ft-scale 用于微调）')
    p.add_argument('--ft-scale', type=float, default=0.1, help='微调学习率缩放系数（默认 0.1）')
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--use-focal', action='store_true', help='使用 FocalLoss（默认关闭）')
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--early-stop', type=int, default=5, help='EarlyStopping patience')
    p.add_argument('--clip-max-norm', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model-name', type=str, default='fine_tuned_model', help='保存名称（保存到 checkpoints/{name}.pth）')
    return p


def main():
    args = build_cli().parse_args()

    print("="*70)
    print("🔄 Transformer Direction Estimator - 增量训练")
    print("="*70)

    if not HAS_TRANSFORMERS:
        print("\n⚠️ 未安装 transformers，将无法使用预训练分词器/Embedding：pip install transformers")
        cont = input("是否继续（将退回字符级Tokenizer，不建议）？[y/N]: ").strip().lower()
        if cont != 'y':
            print("取消")
            return

    if not os.path.exists(args.model):
        print(f"❌ 找不到基础模型: {args.model}")
        return
    if not os.path.exists(args.input):
        print(f"❌ 找不到数据文件: {args.input}")
        return

    cfg = {
        'bert_model': args.bert_model,
        'freeze_embedding': args.freeze_embedding,
        'max_length': args.max_length,
        'train_split': args.train_split,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'ft_scale': args.ft_scale,
        'dropout': args.dropout,
        'use_focal': args.use_focal,
        'focal_gamma': args.focal_gamma,
        'early_stop': args.early_stop,
        'clip_max_norm': args.clip_max_norm,
        'seed': args.seed,
        'model_name': args.model_name,
    }

    print("\n本次微调配置：")
    print("-"*70)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("-"*70)

    start = input("\n开始增量训练？[Y/n]: ").strip().lower()
    if start == 'n':
        print("取消")
        return

    try:
        fine_tune(args.model, args.input, cfg)
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()