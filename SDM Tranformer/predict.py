# predict.py (Transformer Version + ROUTING METRICS)
import os
import argparse
import glob
import time
import numpy as np
import pandas as pd
import torch
import contextlib
from typing import List

from transformer_de import TransformerDirectionEstimator

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def list_model_files(directory='checkpoints') -> List[str]:
    if not os.path.exists(directory):
        return []
    model_files = glob.glob(os.path.join(directory, '*.pth'))
    return sorted(model_files, key=os.path.getmtime, reverse=True)


def select_model_interactive():
    print("\n" + "="*70)
    print("🤖 选择模型文件")
    print("="*70)

    model_files = list_model_files()
    if not model_files:
        print("❌ 未找到模型文件 (在checkpoints/目录)")
        model_path = input("\n请输入模型文件路径: ").strip()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到文件: {model_path}")
        return model_path

    print("\n找到以下模型:")
    for i, file in enumerate(model_files, 1):
        size = os.path.getsize(file) / (1024 * 1024)
        mtime = os.path.getmtime(file)
        time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
        try:
            checkpoint = torch.load(file, map_location='cpu')
            acc = checkpoint.get('val_acc', 0)
            freeze = checkpoint.get('freeze_embedding', False)
            embed_type = "冻结" if freeze else "微调"
            bert_name = checkpoint.get('bert_model_name') or checkpoint.get('config', {}).get('bert_model_name')
            print(f"  {i}. {os.path.basename(file):30s} ({size:.1f} MB, Acc: {acc:.2f}%, {embed_type}, {bert_name}, {time_str})")
        except Exception:
            print(f"  {i}. {os.path.basename(file):30s} ({size:.1f} MB, {time_str})")

    print(f"  {len(model_files)+1}. 手动输入文件路径")

    while True:
        choice = input(f"\n请选择 [1-{len(model_files)+1}]: ").strip()
        if not choice.isdigit():
            print("❌ 请输入数字")
            continue
        choice = int(choice)
        if 1 <= choice <= len(model_files):
            return model_files[choice-1]
        elif choice == len(model_files) + 1:
            model_path = input("请输入模型文件路径: ").strip()
            if not os.path.exists(model_path):
                print(f"❌ 找不到文件: {model_path}")
                continue
            return model_path
        else:
            print(f"❌ 请输入1-{len(model_files)+1}之间的数字")


DIR_MAP_STR2IDX = {
    'left': 0, 'right': 1, 'bidirectional': 2,
    '左': 0, '右': 1, '双向': 2,
    'l': 0, 'r': 1, 'b': 2,
    'L': 0, 'R': 1, 'B': 2,
    '0': 0, '1': 1, '2': 2,
    0: 0, 1: 1, 2: 2,
}
DIR_IDX2NAME_CN = {0: '正向因果(A→B)', 1: '反向因果(A←B)', 2: '双向因果(A↔B)'}

def _label_to_idx_strict(x):
    return DIR_MAP_STR2IDX.get(x, DIR_MAP_STR2IDX.get(str(x).strip(), None))

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


class Predictor:
    def __init__(self, model_path: str, max_length_override: int = None, use_amp: bool = True):
        self.raw_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = self.raw_device
        self.use_amp = (use_amp and self.raw_device == 'cuda')

        ckpt = torch.load(model_path, map_location=self.device)

        bert_name = (ckpt.get('bert_model_name')
                     or ckpt.get('config', {}).get('bert_model_name')
                     or os.environ.get('SDM_BERT_MODEL', 'xlm-roberta-base'))

        self.use_bert_tokenizer = ckpt.get('use_bert_tokenizer', True)
        if self.use_bert_tokenizer:
            if not HAS_TRANSFORMERS:
                raise ImportError("需要安装 transformers: pip install transformers")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
                self.pad_token_id = ckpt.get('pad_token_id', self.tokenizer.pad_token_id)
            except Exception as e:
                raise RuntimeError(f"无法加载分词器 '{bert_name}': {e}")
        else:
            self.char_to_idx = ckpt.get('char_to_idx')
            if not self.char_to_idx:
                raise RuntimeError("Char 模式需要 ckpt 中的 'char_to_idx'。")
            self.pad_token_id = 0

        freeze_embedding = ckpt.get('freeze_embedding', False)
        dropout = ckpt.get('dropout', ckpt.get('config', {}).get('dropout', 0.5))
        
        # Transformer参数
        nhead = ckpt.get('nhead', ckpt.get('config', {}).get('nhead', 4))
        num_layers = ckpt.get('num_layers', ckpt.get('config', {}).get('num_layers', 2))
        dim_feedforward = ckpt.get('dim_feedforward', ckpt.get('config', {}).get('dim_feedforward', 512))
        
        # 显式Attention分析参数（兼容旧模型）
        use_explicit_attention = ckpt.get('use_explicit_attention', ckpt.get('config', {}).get('use_explicit_attention', True))
        explicit_weight = ckpt.get('explicit_weight', ckpt.get('config', {}).get('explicit_weight', 0.3))

        self.model = TransformerDirectionEstimator(
            vocab_size=ckpt.get('vocab_size', 0),
            d_model=ckpt['d_model'],
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            freeze_embedding=freeze_embedding,
            bert_model_name=bert_name,
            use_explicit_attention=use_explicit_attention,
            explicit_weight=explicit_weight
        )
        self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.max_length = (max_length_override
                           if isinstance(max_length_override, int) and max_length_override > 0
                           else ckpt.get('max_length', ckpt.get('config', {}).get('max_length', 64)))

        try:
            vocab_model = getattr(self.model.embedding, 'num_embeddings', None)
            vocab_tok = getattr(self.tokenizer, 'vocab_size', None) if self.use_bert_tokenizer else len(self.char_to_idx)
            if isinstance(vocab_model, int) and isinstance(vocab_tok, int) and vocab_model != vocab_tok:
                print(f"⚠️ 分词器词表({vocab_tok})与模型embedding词表({vocab_model})不一致。推理可能受影响，请确认 tokenizer 与训练时一致。")
        except Exception:
            pass

        self.direction_map = {0: 'left', 1: 'right', 2: 'bidirectional'}
        self.direction_names = DIR_IDX2NAME_CN

        amp_str = "开启" if self.use_amp else "关闭"
        tokenizer_type = "AutoTokenizer" if self.use_bert_tokenizer else "Char"
        embed_type = "冻结" if freeze_embedding else "微调"
        print(f"✓ 模型已加载 (Tokenizer: {tokenizer_type}, Embedding: {embed_type}, 设备: {self.device}, AMP: {amp_str})")
        print(f"✓ 使用 tokenizer: {bert_name}")
        print(f"✓ max_length: {self.max_length}")
        print(f"✓ pad_token_id: {self.pad_token_id}")
        if 'val_acc' in ckpt:
            print(f"✓ 训练时验证准确率: {ckpt['val_acc']:.2f}%")

    def _amp_ctx(self):
        if self.use_amp and self.device == 'cuda':
            return torch.amp.autocast('cuda', dtype=torch.float16, enabled=True)
        return contextlib.nullcontext()

    def _tokenize_one(self, text: str) -> torch.Tensor:
        if self.use_bert_tokenizer:
            enc = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            return enc['input_ids'].squeeze(0).to(torch.long)
        else:
            tokens = [self.char_to_idx.get(ch, 1) for ch in text[:self.max_length]]
            if len(tokens) < self.max_length:
                tokens += [0] * (self.max_length - len(tokens))
            return torch.tensor(tokens, dtype=torch.long)

    def predict_one(self, text: str):
        input_ids = self._tokenize_one(text).unsqueeze(0).to(self.device)
        with torch.no_grad(), self._amp_ctx():
            logits = self.model(input_ids)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        return {
            'text': text,
            'pred_label': self.direction_map[pred_idx],
            'pred_label_cn': self.direction_names[pred_idx],
            'confidence': confidence,
            'prob_left': float(probs[0]),
            'prob_right': float(probs[1]),
            'prob_bidirectional': float(probs[2]),
        }

    def predict_csv(self, input_csv: str, output_csv: str, text_col: str = 'text',
                    label_col: str = 'direction', batch_size: int = 64, verbose: bool = True):
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"找不到输入文件: {input_csv}")

        df = pd.read_csv(input_csv, encoding='utf-8')
        if text_col not in df.columns:
            raise ValueError(f"CSV缺少'{text_col}'列")

        has_label_col = (label_col in df.columns)

        if verbose:
            print(f"\n处理CSV: {input_csv}")
            print(f"  共 {len(df)} 条样本")
            if has_label_col:
                print(f"  包含标签列 '{label_col}'，将计算评估指标")

        texts = df[text_col].astype(str).tolist()
        input_ids_list = [self._tokenize_one(t) for t in texts]
        batch_data = torch.stack(input_ids_list).to(self.device)

        all_probs = []
        total_batches = (len(batch_data) + batch_size - 1) // batch_size
        iterator = range(total_batches)
        if verbose and HAS_TQDM:
            iterator = tqdm(iterator, desc="批量预测")

        with torch.no_grad():
            for i in iterator:
                start = i * batch_size
                end = min(start + batch_size, len(batch_data))
                batch = batch_data[start:end]
                with self._amp_ctx():
                    logits = self.model(batch)
                probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs_batch)

        all_probs = np.vstack(all_probs)
        preds_idx = np.argmax(all_probs, axis=1)
        confidences = all_probs[np.arange(len(all_probs)), preds_idx]

        df['pred_direction'] = [self.direction_map[int(i)] for i in preds_idx]
        df['pred_direction_cn'] = [self.direction_names[int(i)] for i in preds_idx]
        df['confidence'] = confidences
        df['prob_left'] = all_probs[:, 0]
        df['prob_right'] = all_probs[:, 1]
        df['prob_bidirectional'] = all_probs[:, 2]

        if has_label_col:
            labels_raw = df[label_col].tolist()
            labels_idx = [_label_to_idx_strict(lbl) for lbl in labels_raw]
            cm = np.zeros((3, 3), dtype=np.int64)
            valid_eval_count = 0
            for pred, gold in zip(preds_idx, labels_idx):
                if gold is not None and gold in (0, 1, 2):
                    cm[gold, pred] += 1
                    valid_eval_count += 1

        # 路由指标
        p_left = all_probs[:, 0]
        p_right = all_probs[:, 1]
        p_bi = all_probs[:, 2]
        route_final = []
        p_nonleft_list = []
        for j in range(len(preds_idx)):
            is_nonleft = 1 if preds_idx[j] in (1, 2) else 0
            route_final.append('bidir' if is_nonleft else 'forward')
            p_nonleft_list.append(float(p_right[j] + p_bi[j]))

        df['route_final'] = route_final
        df['p_nonleft'] = p_nonleft_list

        if has_label_col:
            gold_route_bin = []
            for lab in labels_idx:
                if lab in (0, 1, 2):
                    gold_route_bin.append(0 if lab == 0 else 1)
                else:
                    gold_route_bin.append(None)

            pred_route_bin = [0 if idx == 0 else 1 for idx in preds_idx]

            cm2 = np.zeros((2, 2), dtype=np.int64)
            valid2 = 0
            for g, p in zip(gold_route_bin, pred_route_bin):
                if g is None:
                    continue
                cm2[g, p] += 1
                valid2 += 1

        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or '.', exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✓ 已保存预测结果: {output_csv} （{len(df)} 条）")

        if has_label_col and valid_eval_count > 0:
            total = cm.sum()
            acc = np.trace(cm) / max(1, total)
            P, R, F1, mP, mR, mF1 = _per_class_metrics(cm)

            print("\n评估指标（跳过未知标签样本）")
            print("-"*70)
            print(f"有效评测样本: {valid_eval_count} / {len(df)}")
            print(f"Overall Acc: {acc*100:.2f}%  |  Macro P/R/F1: {mP:.3f}/{mR:.3f}/{mF1:.3f}")
            print("\n混淆矩阵:")
            header = '实际\\预测'
            col_names = [DIR_IDX2NAME_CN[i] for i in range(3)]
            print(f'{header:12s}  ' + '  '.join([f'{name:12s}' for name in col_names]))
            for i in range(3):
                row = '  '.join([f'{int(v):12d}' for v in cm[i]])
                print(f'{DIR_IDX2NAME_CN[i]:12s}  {row}')
            print("\n按类 P/R/F1:")
            for i, name in enumerate(col_names):
                print(f"  {name}: P={P[i]:.3f} R={R[i]:.3f} F1={F1[i]:.3f}")
            print("-"*70)
        elif has_label_col:
            print("\n⚠️ 没有任何合法标签样本（0/1/2 或 left/right/bidirectional/中文/LRB），跳过三分类指标计算。")

        if has_label_col:
            try:
                if valid2 > 0:
                    total2 = cm2.sum()
                    acc2 = np.trace(cm2) / max(1, total2)
                    P2, R2, F12, mP2, mR2, mF12 = _per_class_metrics(cm2)

                    print("\n[路由层 (forward vs bidir) 指标]")
                    print("-"*70)
                    print(f"有效评测样本(二分类): {valid2} / {len(df)}")
                    print(f"Routing Acc: {acc2*100:.2f}%  |  Macro P/R/F1: {mP2:.3f}/{mR2:.3f}/{mF12:.3f}")
                    print("\n二分类混淆矩阵 (行=真实 gold, 列=预测 pred)")
                    print("            pred: forward    bidir")
                    print(f"gold: forward    {int(cm2[0,0]):10d} {int(cm2[0,1]):10d}")
                    print(f"gold: bidir      {int(cm2[1,0]):10d} {int(cm2[1,1]):10d}")
                    print("\n按类 P/R/F1:")
                    print(f"  forward: P={P2[0]:.3f} R={R2[0]:.3f} F1={F12[0]:.3f}")
                    print(f"  bidir  : P={P2[1]:.3f} R={R2[1]:.3f} F1={F12[1]:.3f}")
                    print("-"*70)
                else:
                    print("\n⚠️ 二分类评估：没有任何合法路由金标，跳过。")
            except NameError:
                pass

        return df, (cm if (has_label_col and valid_eval_count > 0) else None)


def interactive_mode(predictor: Predictor):
    print("\n" + "="*70)
    print("🎯 交互式预测模式")
    print("="*70)
    print("输入文本进行预测，输入 'quit' / 'q' 退出")
    print("-"*70)

    while True:
        text = input("\n📝 请输入文本: ").strip()
        if text.lower() in ['quit', 'q', 'exit', '退出']:
            print("\n👋 再见！")
            break
        if not text:
            print("⚠️ 文本不能为空，请重新输入")
            continue

        try:
            r = predictor.predict_one(text)
            print("\n" + "="*70)
            print("📊 预测结果")
            print("="*70)
            print(f"文本: {text}")
            print(f"\n🎯 预测方向: {r['pred_label_cn']} ({r['pred_label']})")
            print(f"📈 置信度: {r['confidence']:.2%}")
            print("\n详细概率分布：")
            bar = lambda x: '█' * int(x * 30)
            print(f"  正向因果(A→B):   {r['prob_left']:.2%} {bar(r['prob_left'])}")
            print(f"  反向因果(A←B):   {r['prob_right']:.2%} {bar(r['prob_right'])}")
            print(f"  双向因果(A↔B):   {r['prob_bidirectional']:.2%} {bar(r['prob_bidirectional'])}")

            p_nonleft = r['prob_right'] + r['prob_bidirectional']
            final_route = 'bidir' if p_nonleft >= 0.5 else 'forward'
            print(f"\n🧭 最终路由方向(策略)：{final_route}  (p_nonleft={p_nonleft:.2%})")
            print("="*70)
        except Exception as e:
            print(f"\n❌ 预测出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='预测脚本 - 支持模型选择、CSV批量预测和交互式预测（Transformer版本）')
    parser.add_argument('-m', '--model', type=str, help='模型文件路径')
    parser.add_argument('-i', '--input', type=str, nargs='+', help='输入CSV文件（可多个）')
    parser.add_argument('-o', '--output-dir', type=str, default='.', help='输出目录')
    parser.add_argument('-t', '--text-col', type=str, default='text', help='文本列名')
    parser.add_argument('--label-col', type=str, default='direction', help='标签列名（若存在则计算指标）')
    parser.add_argument('--batch-size', type=int, default=64, help='推理批大小')
    parser.add_argument('--max-length', type=int, default=None, help='可覆盖 ckpt 的 max_length')
    parser.add_argument('--interactive', action='store_true', help='交互式预测模式')
    parser.add_argument('--no-amp', action='store_true', help='关闭 AMP（默认开启，CUDA 可用时生效）')

    args = parser.parse_args()

    print("="*70)
    print("🎯 Transformer Direction Estimator - 预测工具")
    print("="*70)

    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ 错误: 找不到模型文件 {model_path}")
            return
    else:
        model_path = select_model_interactive()

    try:
        predictor = Predictor(
            model_path=model_path,
            max_length_override=args.max_length,
            use_amp=(not args.no_amp)
        )
    except Exception as e:
        print(f"\n❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return

    if not args.interactive:
        if args.input:
            csv_paths = args.input
            for p in csv_paths:
                if not os.path.exists(p):
                    print(f"❌ 错误: 找不到文件 {p}")
                    return
        else:
            csv_paths = []

        if csv_paths:
            os.makedirs(args.output_dir, exist_ok=True)
            for csv_path in csv_paths:
                out_name = os.path.basename(csv_path).replace('.csv', '_predicted.csv')
                out_path = os.path.join(args.output_dir, out_name)
                try:
                    predictor.predict_csv(
                        input_csv=csv_path,
                        output_csv=out_path,
                        text_col=args.text_col,
                        label_col=args.label_col,
                        batch_size=args.batch_size,
                        verbose=True
                    )
                except Exception as e:
                    print(f"\n❌ 处理 {csv_path} 时出错: {e}")
                    import traceback
                    traceback.print_exc()

    if args.interactive or not args.input:
        interactive_mode(predictor)


if __name__ == '__main__':
    main()