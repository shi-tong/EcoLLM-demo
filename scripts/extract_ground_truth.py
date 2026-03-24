#!/usr/bin/env python3
"""
从测试数据中提取 Ground Truth
将对话中的 record_process_flow 调用提取为标准格式，用于评估
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def extract_records_from_dialogue(dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从一个对话中提取所有的 record_process_flow 调用"""
    records = []
    messages = dialogue.get('messages', [])
    
    for msg in messages:
        if msg.get('role') == 'assistant':
            tool_calls = msg.get('tool_calls', [])
            
            for tc in tool_calls:
                if tc.get('name') == 'record_process_flow':
                    args = tc.get('arguments', {})
                    
                    # 提取关键字段
                    record = {
                        'flow_name': args.get('name', ''),
                        'category': args.get('category', ''),
                        'value': args.get('value', 0),
                        'unit': args.get('unit', ''),
                        'flow_type': args.get('flow_type', ''),
                    }
                    
                    # 提取 chunk_id（如果有）
                    selected_chunk = args.get('selected_chunk', {})
                    if isinstance(selected_chunk, dict):
                        record['chunk_id'] = selected_chunk.get('chunk_id', '')
                    else:
                        record['chunk_id'] = ''
                    
                    # 提取 note（如果有）
                    if 'note' in args:
                        record['note'] = args['note']
                    
                    records.append(record)
    
    return records


def process_test_files(test_dir: Path) -> List[Dict[str, Any]]:
    """处理所有测试文件，提取 ground truth"""
    ground_truth = []
    
    # 处理 FULL 对话
    full_files = list(test_dir.glob('full/*_with_think.json'))
    for filepath in sorted(full_files):
        print(f"处理 FULL: {filepath.name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理可能的列表格式
        dialogues = data if isinstance(data, list) else [data]
        
        for dialogue in dialogues:
            records = extract_records_from_dialogue(dialogue)
            ground_truth.append({
                'file': filepath.name,
                'type': 'FULL',
                'records': records
            })
    
    # 处理 SHORT_E 对话
    short_e_files = list(test_dir.glob('short_extract/*_complete.json'))
    for filepath in sorted(short_e_files):
        print(f"处理 SHORT_E: {filepath.name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dialogues = data if isinstance(data, list) else [data]
        
        for dialogue in dialogues:
            records = extract_records_from_dialogue(dialogue)
            ground_truth.append({
                'file': filepath.name,
                'type': 'SHORT_E',
                'records': records
            })
    
    # 处理 SHORT_QA 对话（通常没有 record，但检查一下）
    short_qa_files = list(test_dir.glob('short_qa/*_complete.json'))
    for filepath in sorted(short_qa_files):
        print(f"处理 SHORT_QA: {filepath.name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dialogues = data if isinstance(data, list) else [data]
        
        for dialogue in dialogues:
            records = extract_records_from_dialogue(dialogue)
            # SHORT_QA 可能没有 records，但仍然记录
            ground_truth.append({
                'file': filepath.name,
                'type': 'SHORT_QA',
                'records': records
            })
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="从测试数据中提取 Ground Truth")
    parser.add_argument(
        '--test-dir',
        type=str,
        default='test_data',
        help='测试数据目录（默认: test_data）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_data/ground_truth.json',
        help='输出文件路径（默认: test_data/ground_truth.json）'
    )
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"❌ 错误：测试数据目录不存在: {test_dir}")
        return
    
    print(f"📂 从 {test_dir} 提取 Ground Truth...")
    ground_truth = process_test_files(test_dir)
    
    # 统计信息
    total_records = sum(len(item['records']) for item in ground_truth)
    full_count = sum(1 for item in ground_truth if item['type'] == 'FULL')
    short_e_count = sum(1 for item in ground_truth if item['type'] == 'SHORT_E')
    short_qa_count = sum(1 for item in ground_truth if item['type'] == 'SHORT_QA')
    
    print(f"\n📊 统计:")
    print(f"  - FULL 对话: {full_count}")
    print(f"  - SHORT_E 对话: {short_e_count}")
    print(f"  - SHORT_QA 对话: {short_qa_count}")
    print(f"  - 总对话数: {len(ground_truth)}")
    print(f"  - 总记录数: {total_records}")
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Ground Truth 已保存到: {output_path}")
    
    # 显示示例
    if ground_truth and ground_truth[0]['records']:
        print(f"\n📝 示例记录（来自 {ground_truth[0]['file']}）:")
        for i, record in enumerate(ground_truth[0]['records'][:3], 1):
            print(f"  {i}. {record['flow_name']}: {record['value']} {record['unit']} ({record['category']})")


if __name__ == '__main__':
    main()
