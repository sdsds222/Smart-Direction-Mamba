"""
增强版评估脚本 - 支持选择模型文件和多个CSV文件
"""

import torch
import pandas as pd
from mamba_de import MambaDirectionEstimator
from tqdm import tqdm
import argparse
import os
import numpy as np
import glob

try:
    from transformers import BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def list_model_files(directory='checkpoints'):
    """列出所有模型文件"""
    if not os.path.exists(directory):
        return []
    
    model_files = glob.glob(os.path.join(directory, '*.pth'))
    return sorted(model_files, key=os.path.getmtime, reverse=True)  # 按时间排序


def list_csv_files(directory='.'):
    """列出所有CSV文件"""
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    return sorted(csv_files)


def select_model_interactive():
    """交互式选择模型文件"""
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
        size = os.path.getsize(file) / (1024 * 1024)  # MB
        mtime = os.path.getmtime(file)
        import time
        time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
        
        # 尝试读取准确率
        try:
            checkpoint = torch.load(file, map_location='cpu')
            acc = checkpoint.get('val_acc', 0)
            print(f"  {i}. {os.path.basename(file):30s} ({size:.1f} MB, 准确率: {acc:.2f}%, {time_str})")
        except:
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


def select_multiple_csv(csv_files):
    """选择多个CSV文件"""
    print("\n" + "="*70)
    print("📚 多CSV文件选择")
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
        selection = input(f"\n请选择要预测的CSV文件: ").strip()
        
        if selection.lower() == 'all':
            return csv_files
        
        try:
            if ',' in selection:
                indices = [int(x.strip()) for x in selection.split(',')]
            else:
                indices = [int(x.strip()) for x in selection.split()]
            
            selected_files = []
            for idx in indices:
                if 1 <= idx <= len(csv_files):
                    selected_files.append(csv_files[idx-1])
                else:
                    print(f"❌ 序号 {idx} 无效")
                    break
            else:
                if selected_files:
                    print(f"\n✓ 已选择 {len(selected_files)} 个文件:")
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
    print("📁 选择评估数据")
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
    print(f"  {len(csv_files)+2}. 选择多个CSV文件")
    
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


class Evaluator:
    """评估器"""
    
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.use_bert_tokenizer = checkpoint.get('use_bert_tokenizer', True)
        
        if self.use_bert_tokenizer:
            if not HAS_TRANSFORMERS:
                raise ImportError("需要安装: pip install transformers")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        else:
            self.char_to_idx = checkpoint['char_to_idx']
        
        self.model = MambaDirectionEstimator(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            d_state=checkpoint['d_state']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.max_length = 64
        
        self.direction_map = {0: 'left', 1: 'right', 2: 'bidirectional'}
        self.direction_names = {0: '左向(因果)', 1: '右向(反因果)', 2: '双向'}
        
        self.label_to_idx = {
            'left': 0, 'right': 1, 'bidirectional': 2,
            '左': 0, '右': 1, '双向': 2,
            'L': 0, 'R': 1, 'B': 2
        }
        
        tokenizer_type = "BERT" if self.use_bert_tokenizer else "字符级"
        print(f"✓ 模型已加载 (Tokenizer: {tokenizer_type}, 设备: {self.device})")
        print(f"✓ 验证准确率: {checkpoint['val_acc']:.2f}%")
    
    def tokenize(self, text: str):
        if self.use_bert_tokenizer:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            return torch.tensor([encoded], dtype=torch.long)
        else:
            tokens = [self.char_to_idx.get(char, 1) for char in text[:self.max_length]]
            if len(tokens) < self.max_length:
                tokens = tokens + [0] * (self.max_length - len(tokens))
            return torch.tensor([tokens], dtype=torch.long)
    
    def predict_one(self, text):
        input_ids = self.tokenize(text).to(self.device)
        
        with torch.no_grad():
            directions, probs = self.model.predict(input_ids)
        
        pred_idx = directions[0].item()
        
        return {
            'pred_idx': pred_idx,
            'pred_label': self.direction_map[pred_idx],
            'pred_label_cn': self.direction_names[pred_idx],
            'confidence': float(probs[0][pred_idx]),
            'prob_left': float(probs[0][0]),
            'prob_right': float(probs[0][1]),
            'prob_bi': float(probs[0][2])
        }
    
    def evaluate_csv(self, input_path, output_path, text_col='text', label_col='direction'):
        """评估单个CSV文件"""
        print(f"\n{'='*70}")
        print(f"评估文件: {os.path.basename(input_path)}")
        print(f"{'='*70}")
        
        df = pd.read_csv(input_path, encoding='utf-8')
        
        if text_col not in df.columns:
            raise ValueError(f"找不到文本列 '{text_col}'")
        
        has_labels = label_col in df.columns
        
        print(f"✓ 加载了 {len(df)} 条样本")
        
        if has_labels:
            true_labels_idx = []
            true_labels_cn = []
            
            for label in df[label_col]:
                label_str = str(label).strip()
                true_idx = self.label_to_idx.get(label_str, -1)
                if true_idx == -1:
                    true_labels_cn.append("未知")
                else:
                    true_labels_cn.append(self.direction_names[true_idx])
                true_labels_idx.append(true_idx)
            
            df['true_label_idx'] = true_labels_idx
            df['true_label_cn'] = true_labels_cn
            
            valid_df = df[df['true_label_idx'] != -1].copy()
            if len(valid_df) < len(df):
                print(f"⚠️  过滤掉 {len(df) - len(valid_df)} 条无效标签")
            
            print(f"✓ 有效样本: {len(valid_df)} 条")
        else:
            print("ℹ️  无标签列，仅进行预测")
            valid_df = df.copy()
        
        print("\n开始预测...")
        predictions = []
        
        for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="预测进度"):
            text = str(row[text_col])
            result = self.predict_one(text)
            predictions.append(result)
        
        valid_df['pred_label'] = [p['pred_label'] for p in predictions]
        valid_df['pred_label_cn'] = [p['pred_label_cn'] for p in predictions]
        valid_df['pred_idx'] = [p['pred_idx'] for p in predictions]
        valid_df['confidence'] = [p['confidence'] for p in predictions]
        valid_df['prob_left'] = [p['prob_left'] for p in predictions]
        valid_df['prob_right'] = [p['prob_right'] for p in predictions]
        valid_df['prob_bidirectional'] = [p['prob_bi'] for p in predictions]
        
        if has_labels:
            valid_df['correct'] = valid_df['true_label_idx'] == valid_df['pred_idx']
            valid_df['result'] = valid_df['correct'].apply(lambda x: '✓ 正确' if x else '✗ 错误')
        
        valid_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ 结果已保存到: {output_path}")
        
        if has_labels:
            self.print_evaluation(valid_df)
        else:
            self.print_prediction_summary(valid_df)
        
        return valid_df
    
    def print_evaluation(self, df):
        """打印评估结果（有标签）"""
        print("\n" + "="*70)
        print("评估结果")
        print("="*70)
        
        accuracy = (df['correct'].sum() / len(df)) * 100
        print(f"\n总体准确率: {accuracy:.2f}% ({df['correct'].sum()}/{len(df)})")
        
        print(f"平均置信度: {df['confidence'].mean():.2%}")
        if df['correct'].sum() > 0:
            print(f"  正确预测的置信度: {df[df['correct']]['confidence'].mean():.2%}")
        if (~df['correct']).sum() > 0:
            print(f"  错误预测的置信度: {df[~df['correct']]['confidence'].mean():.2%}")
        
        print("\n方向分布:")
        direction_counts = df['pred_label_cn'].value_counts()
        for direction, count in direction_counts.items():
            pct = 100 * count / len(df)
            print(f"  {direction}: {count} ({pct:.1f}%)")
    
    def print_prediction_summary(self, df):
        """打印预测摘要（无标签）"""
        print("\n" + "="*70)
        print("预测结果摘要")
        print("="*70)
        
        print(f"\n总样本数: {len(df)}")
        print(f"平均置信度: {df['confidence'].mean():.2%}")
        
        print("\n预测方向分布:")
        direction_counts = df['pred_label_cn'].value_counts()
        for direction, count in direction_counts.items():
            pct = 100 * count / len(df)
            print(f"  {direction}: {count} ({pct:.1f}%)")


def interactive_mode(evaluator):
    """交互模式"""
    print("\n" + "="*70)
    print("🎯 交互式预测模式")
    print("="*70)
    print("输入文本进行预测，输入 'quit' 或 'q' 退出")
    print("-"*70)
    
    while True:
        text = input("\n📝 请输入文本: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q', '退出']:
            print("\n👋 再见！")
            break
        
        if not text:
            print("⚠️  文本不能为空，请重新输入")
            continue
        
        try:
            result = evaluator.predict_one(text)
            
            print("\n" + "="*70)
            print("📊 预测结果")
            print("="*70)
            print(f"文本: {text}")
            print(f"\n🎯 预测方向: {result['pred_label_cn']} ({result['pred_label']})")
            print(f"📈 置信度: {result['confidence']:.2%}")
            print(f"\n详细概率分布:")
            print(f"  左向(因果):      {result['prob_left']:.2%} {'█' * int(result['prob_left'] * 30)}")
            print(f"  右向(反因果):    {result['prob_right']:.2%} {'█' * int(result['prob_right'] * 30)}")
            print(f"  双向:            {result['prob_bi']:.2%} {'█' * int(result['prob_bi'] * 30)}")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ 预测出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='增强版评估 - 支持选择模型和多CSV文件')
    parser.add_argument('-m', '--model', type=str, help='模型文件路径')
    parser.add_argument('-i', '--input', type=str, nargs='+', help='输入CSV文件（可多个）')
    parser.add_argument('-o', '--output-dir', type=str, default='.', help='输出目录')
    parser.add_argument('-t', '--text-col', type=str, default='text', help='文本列名')
    parser.add_argument('-l', '--label-col', type=str, default='direction', help='标签列名')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎯 Mamba Direction Estimator - 增强版评估工具")
    print("="*70)
    
    # 1. 选择模型
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ 错误: 找不到模型文件 {model_path}")
            return
    else:
        model_path = select_model_interactive()
    
    # 2. 加载模型
    try:
        evaluator = Evaluator(model_path)
    except Exception as e:
        print(f"\n❌ 加载模型失败: {e}")
        return
    
    # 3. 选择CSV文件（如果不是交互模式）
    if not args.interactive:
        if args.input:
            csv_paths = args.input
            for csv_path in csv_paths:
                if not os.path.exists(csv_path):
                    print(f"❌ 错误: 找不到文件 {csv_path}")
                    return
        else:
            csv_paths = select_csv_interactive()
        
        # 4. 批量评估
        os.makedirs(args.output_dir, exist_ok=True)
        
        all_results = []
        for csv_path in csv_paths:
            output_name = os.path.basename(csv_path).replace('.csv', '_predicted.csv')
            output_path = os.path.join(args.output_dir, output_name)
            
            try:
                df = evaluator.evaluate_csv(
                    input_path=csv_path,
                    output_path=output_path,
                    text_col=args.text_col,
                    label_col=args.label_col
                )
                all_results.append({
                    'file': os.path.basename(csv_path),
                    'samples': len(df),
                    'output': output_path
                })
            except Exception as e:
                print(f"\n❌ 处理 {csv_path} 时出错: {e}")
                continue
        
        # 5. 总结
        if len(csv_paths) > 1:
            print("\n" + "="*70)
            print("📊 批量评估完成")
            print("="*70)
            for result in all_results:
                print(f"\n文件: {result['file']}")
                print(f"  样本数: {result['samples']}")
                print(f"  输出: {result['output']}")
    
    # 6. 交互模式
    if args.interactive or not args.input:
        interactive_mode(evaluator)


if __name__ == '__main__':
    main()