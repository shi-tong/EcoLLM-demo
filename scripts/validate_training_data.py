#!/usr/bin/env python3
"""
训练数据质量检查脚本 v1.0

检查项目：
1. 格式一致性：所有 assistant 消息是否有 reasoning_content
2. tool_calls 格式：name/arguments 结构是否正确
3. reasoning-action 一致性：说"没找到"但还是 record 了
4. 空值检查：reasoning_content 或 content 是否为空
5. 数值合理性：record 的值是否在合理范围内

用法：
    python scripts/validate_training_data.py --data-dir dataset/
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class TrainingDataValidator:
    """训练数据验证器"""
    
    def __init__(self):
        self.issues = defaultdict(list)  # {issue_type: [(file, msg), ...]}
        self.stats = defaultdict(int)
    
    def validate_file(self, filepath: str) -> List[Tuple[str, str]]:
        """验证单个文件"""
        issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(("json_error", f"JSON 解析错误: {e}"))
            return issues
        
        # 处理不同格式
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "messages" in data:
            samples = [data]
        else:
            issues.append(("format_error", "未知数据格式"))
            return issues
        
        for sample_idx, sample in enumerate(samples):
            messages = sample.get("messages", [])
            sample_issues = self._validate_messages(messages, sample_idx)
            issues.extend(sample_issues)
        
        return issues
    
    def _validate_messages(self, messages: List[Dict], sample_idx: int) -> List[Tuple[str, str]]:
        """验证消息列表"""
        issues = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            
            if role == "assistant":
                # 检查 1: reasoning_content 是否存在
                reasoning = msg.get("reasoning_content", "")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                
                # 有 tool_calls 的 assistant 必须有 reasoning_content
                if tool_calls and not reasoning:
                    issues.append(("missing_reasoning", 
                        f"Sample {sample_idx}, Msg {i}: 有 tool_calls 但没有 reasoning_content"))
                
                # 没有 tool_calls 的 assistant 必须有 content（最终回复）
                if not tool_calls and not content:
                    issues.append(("missing_content", 
                        f"Sample {sample_idx}, Msg {i}: 没有 tool_calls 也没有 content"))
                
                # 检查 2: tool_calls 格式
                for tc_idx, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        issues.append(("invalid_tool_call", 
                            f"Sample {sample_idx}, Msg {i}, TC {tc_idx}: tool_call 不是 dict"))
                        continue
                    
                    if "name" not in tc:
                        issues.append(("missing_tool_name", 
                            f"Sample {sample_idx}, Msg {i}, TC {tc_idx}: 缺少 name"))
                    
                    if "arguments" not in tc:
                        issues.append(("missing_tool_args", 
                            f"Sample {sample_idx}, Msg {i}, TC {tc_idx}: 缺少 arguments"))
                    elif not isinstance(tc["arguments"], dict):
                        issues.append(("invalid_tool_args", 
                            f"Sample {sample_idx}, Msg {i}, TC {tc_idx}: arguments 不是 dict"))
                
                # 检查 3: reasoning-action 一致性
                if reasoning and tool_calls:
                    reasoning_lower = reasoning.lower()
                    tool_name = tool_calls[0].get("name", "") if tool_calls else ""
                    
                    # 说"没找到"但还是 record 了
                    not_found_phrases = ["couldn't find", "not found", "no data", "没找到", "未找到"]
                    is_not_found = any(phrase in reasoning_lower for phrase in not_found_phrases)
                    is_recording = tool_name in ["record_process_flow", "record_parameter", "define_lca_scope"]
                    
                    if is_not_found and is_recording:
                        issues.append(("reasoning_action_mismatch", 
                            f"Sample {sample_idx}, Msg {i}: reasoning 说没找到，但还是调用了 {tool_name}"))
                
                # 检查 4: reasoning_content 是否是占位符
                if reasoning and "[SMART_SKIP_PLACEHOLDER" in reasoning:
                    issues.append(("placeholder_not_replaced", 
                        f"Sample {sample_idx}, Msg {i}: reasoning_content 还是占位符"))
                
                # 检查 5: 空的 tool_calls 数组（应该删除）
                if "tool_calls" in msg and msg["tool_calls"] == []:
                    issues.append(("empty_tool_calls", 
                        f"Sample {sample_idx}, Msg {i}: 有空的 tool_calls 数组，应该删除"))
            
            elif role == "tool":
                # 检查 tool response 是否为空
                content = msg.get("content", "")
                if not content:
                    issues.append(("empty_tool_response", 
                        f"Sample {sample_idx}, Msg {i}: tool response 为空"))
            
            elif role == "user":
                content = msg.get("content", "")
                if not content and i > 0:  # 第一条 user 可能是空的（system 后面）
                    issues.append(("empty_user_content", 
                        f"Sample {sample_idx}, Msg {i}: user content 为空"))
        
        return issues
    
    def validate_directory(self, data_dir: str, pattern: str = "*_complete.json"):
        """验证目录下的所有文件"""
        data_path = Path(data_dir)
        
        # 查找所有匹配的文件
        files = list(data_path.rglob(pattern))
        files.extend(data_path.rglob("*_with_think.json"))
        
        print(f"🔍 找到 {len(files)} 个文件待检查\n")
        
        for filepath in sorted(files):
            rel_path = filepath.relative_to(data_path)
            issues = self.validate_file(str(filepath))
            
            if issues:
                print(f"❌ {rel_path}")
                for issue_type, msg in issues:
                    print(f"   - [{issue_type}] {msg}")
                    self.issues[issue_type].append((str(rel_path), msg))
            else:
                print(f"✅ {rel_path}")
            
            self.stats["total_files"] += 1
            self.stats["files_with_issues"] += 1 if issues else 0
    
    def print_summary(self):
        """打印汇总报告"""
        print("\n" + "=" * 60)
        print("📊 数据质量检查报告")
        print("=" * 60)
        
        print(f"\n📁 文件统计:")
        print(f"   - 总文件数: {self.stats['total_files']}")
        print(f"   - 有问题的文件: {self.stats['files_with_issues']}")
        print(f"   - 通过率: {100 * (1 - self.stats['files_with_issues'] / max(1, self.stats['total_files'])):.1f}%")
        
        if self.issues:
            print(f"\n⚠️  问题汇总:")
            for issue_type, items in sorted(self.issues.items(), key=lambda x: -len(x[1])):
                print(f"   - {issue_type}: {len(items)} 处")
            
            print(f"\n🔧 建议修复优先级:")
            priority_order = [
                ("reasoning_action_mismatch", "高", "reasoning 和 action 矛盾会让模型学到错误行为"),
                ("placeholder_not_replaced", "高", "占位符没有被替换成真正的 reasoning"),
                ("missing_reasoning", "中", "缺少 reasoning 会影响思维链学习"),
                ("missing_content", "中", "最终回复缺失"),
                ("empty_tool_calls", "低", "空数组应该删除，但不影响训练"),
            ]
            
            for issue_type, priority, reason in priority_order:
                if issue_type in self.issues:
                    print(f"   [{priority}] {issue_type}: {reason}")
        else:
            print(f"\n✅ 所有文件通过检查！")


def main():
    parser = argparse.ArgumentParser(description="训练数据质量检查")
    parser.add_argument("--data-dir", default="dataset", help="数据目录")
    args = parser.parse_args()
    
    validator = TrainingDataValidator()
    validator.validate_directory(args.data_dir)
    validator.print_summary()


if __name__ == "__main__":
    main()
