#!/usr/bin/env python3
"""
准备微调数据：合并所有训练数据为单个 JSONL 文件
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import random

def load_json_file(file_path: Path) -> List[Dict]:
    """加载 JSON 文件，返回 messages 列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理不同格式
    if isinstance(data, list):
        if len(data) > 0 and 'messages' in data[0]:
            return [item['messages'] for item in data]
        return data
    elif isinstance(data, dict) and 'messages' in data:
        return [data['messages']]
    return []

def collect_data(data_dirs: List[str], patterns: List[str]) -> List[Dict]:
    """收集所有数据，标记数据类型"""
    all_samples = []
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"⚠️  目录不存在: {data_dir}")
            continue
        
        # 判断数据类型
        if "full" in data_dir:
            data_type = "full"
        elif "short_extract" in data_dir:
            data_type = "short_extract"
        elif "short_qa" in data_dir:
            data_type = "short_qa"
        else:
            data_type = "unknown"
        
        for pattern in patterns:
            for file_path in data_path.rglob(pattern):
                try:
                    messages_list = load_json_file(file_path)
                    for messages in messages_list:
                        all_samples.append({
                            "messages": messages,
                            "source": str(file_path),
                            "type": data_type
                        })
                except Exception as e:
                    print(f"⚠️  加载失败 {file_path}: {e}")
    
    return all_samples

def clean_messages(messages: List[Dict]) -> List[Dict]:
    """清理 messages，移除不需要的字段"""
    cleaned = []
    for msg in messages:
        clean_msg = {
            "role": msg["role"],
        }
        
        # 保留 content
        if "content" in msg:
            clean_msg["content"] = msg["content"]
        
        # 保留 reasoning_content（用于 reasoning 训练）
        if "reasoning_content" in msg and msg["reasoning_content"]:
            clean_msg["reasoning_content"] = msg["reasoning_content"]
        
        # 保留 tool_calls
        if "tool_calls" in msg and msg["tool_calls"]:
            clean_msg["tool_calls"] = msg["tool_calls"]
        
        cleaned.append(clean_msg)
    
    return cleaned

def main():
    parser = argparse.ArgumentParser(description="准备微调数据")
    parser.add_argument("--output", type=str, default="dataset/finetune_data.jsonl",
                        help="输出文件路径")
    parser.add_argument("--eval-full", type=int, default=3,
                        help="验证集中 full 样本数 (默认 3)")
    parser.add_argument("--eval-short", type=int, default=9,
                        help="验证集中 short 样本数 (默认 9)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()
    
    print("=" * 60)
    print("微调数据准备工具")
    print("=" * 60)
    
    # 收集数据
    data_dirs = [
        "dataset/full",
        "dataset/short_extract", 
        "dataset/short_qa"
    ]
    patterns = ["*_with_think.json", "*_complete.json"]
    
    print("\n📂 收集数据...")
    all_samples = collect_data(data_dirs, patterns)
    print(f"   总样本数: {len(all_samples)}")
    
    # 按类型分组
    full_samples = [s for s in all_samples if s["type"] == "full"]
    short_extract_samples = [s for s in all_samples if s["type"] == "short_extract"]
    short_qa_samples = [s for s in all_samples if s["type"] == "short_qa"]
    
    print("\n📊 数据来源分布:")
    print(f"   - full: {len(full_samples)}")
    print(f"   - short_extract: {len(short_extract_samples)}")
    print(f"   - short_qa: {len(short_qa_samples)}")
    
    # 打乱各组
    random.seed(args.seed)
    random.shuffle(full_samples)
    random.shuffle(short_extract_samples)
    random.shuffle(short_qa_samples)
    
    # 分层采样验证集
    eval_full = full_samples[:args.eval_full]
    eval_short_extract = short_extract_samples[:args.eval_short // 2]
    eval_short_qa = short_qa_samples[:args.eval_short - args.eval_short // 2]
    
    train_full = full_samples[args.eval_full:]
    train_short_extract = short_extract_samples[args.eval_short // 2:]
    train_short_qa = short_qa_samples[args.eval_short - args.eval_short // 2:]
    
    # 合并
    train_samples = train_full + train_short_extract + train_short_qa
    eval_samples = eval_full + eval_short_extract + eval_short_qa
    
    random.shuffle(train_samples)
    random.shuffle(eval_samples)
    
    print(f"\n📊 数据集分割 (分层采样):")
    print(f"   - 训练集: {len(train_samples)} 样本 (full: {len(train_full)}, short: {len(train_short_extract) + len(train_short_qa)})")
    print(f"   - 验证集: {len(eval_samples)} 样本 (full: {len(eval_full)}, short: {len(eval_short_extract) + len(eval_short_qa)})")
    
    # 保存训练集
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path.with_suffix('.train.jsonl')
    eval_path = output_path.with_suffix('.eval.jsonl')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            cleaned = clean_messages(sample["messages"])
            f.write(json.dumps({"messages": cleaned}, ensure_ascii=False) + '\n')
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        for sample in eval_samples:
            cleaned = clean_messages(sample["messages"])
            f.write(json.dumps({"messages": cleaned}, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 数据已保存:")
    print(f"   - 训练集: {train_path}")
    print(f"   - 验证集: {eval_path}")
    
    # 显示样本统计
    print("\n📈 Token 统计提示:")
    print("   运行以下命令查看 token 分布:")
    print(f"   python scripts/analyze_full_token_lengths.py --data-dir dataset --pattern '*.train.jsonl'")

if __name__ == "__main__":
    main()
