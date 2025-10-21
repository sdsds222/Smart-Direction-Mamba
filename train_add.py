"""
增量训练脚本 - 在已有模型基础上继续训练 (已优化保存逻辑)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import os
from train import DirectionDatasetBERT#, list_model_files, select_model_interactive
from mamba_de import MambaDirectionEstimator


def fine_tune_model(base_model_path, new_data_path, config):
    """增量训练"""
    print("\n" + "="*70)
    print("🔄 增量训练 (Fine-tuning)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ 使用设备: {device}")
    
    # 1. 加载已有模型
    print(f"\n[1/5] 加载基础模型: {os.path.basename(base_model_path)}")
    checkpoint = torch.load(base_model_path, map_location=device)
    
    model = MambaDirectionEstimator(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        d_state=checkpoint['d_state']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    original_val_acc = checkpoint.get('val_acc', 0.0) # 使用 .get 避免旧模型没这个key
    print(f"✓ 基础模型准确率: {original_val_acc:.2f}%")
    
    # 2. 加载新数据
    print(f"\n[2/5] 加载新数据...")
    dataset = DirectionDatasetBERT(
        csv_path=new_data_path,
        max_length=config['max_length'],
        use_bert_tokenizer=True
    )
    
    if len(dataset) < 10:
        print(f"⚠️  警告: 新数据量过少 ({len(dataset)}条)，可能无法有效划分训练/验证集。")
        train_size = len(dataset)
        val_size = 0
    else:
        train_size = int(config['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    # 如果没有验证集，val_loader为空
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False) if val_size > 0 else None
    
    print(f"✓ 新训练集: {train_size} 样本")
    print(f"✓ 新验证集: {val_size} 样本")
    
    # 3. 设置优化器（使用更小的学习率）
    print(f"\n[3/5] 配置优化器...")
    fine_tune_lr = config['learning_rate'] * 0.1  # 微调用更小的学习率
    print(f"✓ 微调学习率: {fine_tune_lr} (原始: {config['learning_rate']})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    criterion = nn.CrossEntropyLoss()
    
    # 4. 增量训练
    print(f"\n[4/5] 开始增量训练...")
    print("-"*70)
    
    best_finetune_acc = -1.0
    saved_model_path = ""
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
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
        
        val_acc = 0.0
        if val_loader:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
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
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        else:
            val_acc = train_acc
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | (无验证集)')

        
        if val_acc > best_finetune_acc:
            best_finetune_acc = val_acc
            
            model_path_relative = f'checkpoints/{config["model_name"]}.pth'
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'val_acc': val_acc,
                'vocab_size': model.embedding.num_embeddings, 'd_model': model.d_model,
                'd_state': model.d_state, 'use_bert_tokenizer': dataset.use_bert_tokenizer,
                'config': config, 'base_model': base_model_path, 'fine_tuned': True
            }, model_path_relative)
            
            # 【重要改动】: 获取并保存文件的绝对路径
            saved_model_path = os.path.abspath(model_path_relative)
            print(f'  ✓ 保存当前最佳模型 (验证准确率: {val_acc:.2f}%)')
        
        scheduler.step()
    
    # 5. 总结
    print(f"\n[5/5] 增量训练完成!")
    print("="*70)
    print(f"原始模型准确率: {original_val_acc:.2f}%")
    print(f"微调后最佳准确率: {best_finetune_acc:.2f}%")
    
    if best_finetune_acc > -1.0:
        improvement = best_finetune_acc - original_val_acc
        if improvement > 0.01:
             print(f"✅ 提升: +{improvement:.2f}%")
        elif improvement < -0.01:
             print(f"⚠️  下降: {improvement:.2f}% (可能新数据与原数据差异大)")
        else:
             print(f"→ 持平")
        # 【重要改动】: 打印绝对路径
        print(f"\n✓ 微调模型已保存至: {saved_model_path}")
    else:
        print("\n✗ 本次微调未产生有效模型。")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='增量训练 - 在已有模型基础上继续训练')
    parser.add_argument('-m', '--model', type=str, required=True, help='基础模型路径')
    parser.add_argument('-i', '--input', type=str, required=True, help='新训练数据CSV')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数（可适当增加）')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率（会自动缩小10倍）')
    parser.add_argument('--model-name', type=str, default='fine_tuned_model', help='保存名称')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔄 Mamba Direction Estimator - 增量训练")
    print("="*70)
    print("\n💡 增量训练说明:")
    print("  - 在已有模型基础上继续学习新数据")
    print("  - 使用更小的学习率（原始的10%）")
    print("  - 保留原有知识，学习新模式")
    print("="*70)
    
    base_model_path = args.model
    if not os.path.exists(base_model_path):
        print(f"❌ 错误: 找不到模型 {base_model_path}")
        return
    
    if not os.path.exists(args.input):
        print(f"❌ 错误: 找不到数据 {args.input}")
        return
    
    print(f"\n基础模型: {os.path.basename(base_model_path)}")
    print(f"新数据: {os.path.basename(args.input)}")
    
    confirm = input("\n是否开始增量训练? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("取消")
        return
    
    try:
        base_checkpoint = torch.load(base_model_path, map_location='cpu')
        config = base_checkpoint['config']
        config['batch_size'] = args.batch_size
        config['num_epochs'] = args.epochs
        config['learning_rate'] = args.lr
        config['model_name'] = args.model_name
        print("\n✓ 已从基础模型加载配置，并使用命令行参数更新。")
    except Exception as e:
        print(f"⚠️ 无法从旧模型加载配置 ({e})，将使用默认值。")
        config = {
            'max_length': 64, 'train_split': 0.8, 'batch_size': args.batch_size,
            'num_epochs': args.epochs, 'learning_rate': args.lr, 'model_name': args.model_name
        }

    try:
        fine_tune_model(base_model_path, args.input, config)
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

