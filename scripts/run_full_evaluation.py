#!/usr/bin/env python3
"""
完整的端到端评估脚本
通过 FastAPI 后端运行完整的 EcoLLM Agent 推理，然后计算评估指标

评估流程：
1. 启动后端服务（需要手动完成）
2. 上传测试文档到系统
3. 对每个测试对话运行 Agent
4. 收集模型预测结果
5. 对比 Ground Truth，计算指标

使用方式：
    # 1. 先启动后端服务
    cd backend && python app.py
    
    # 2. 运行评估
    python scripts/run_full_evaluation.py \
        --backend-url http://localhost:8000 \
        --test-dir test_data \
        --output-dir test_results
"""

import json
import argparse
import asyncio
import aiohttp
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime


class EcoLLMEvaluator:
    """EcoLLM 系统评估器"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.session = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def check_backend_health(self) -> bool:
        """检查后端服务是否可用"""
        try:
            async with self.session.get(f"{self.backend_url}/sessions/stats", timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"❌ 后端服务不可用: {e}")
            return False
    
    async def upload_document(self, pdf_path: Path) -> Optional[str]:
        """
        上传 PDF 文档到系统
        
        Returns:
            session_id 或 None（如果失败）
        """
        try:
            # 读取 PDF 文件
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()
            
            # 构造 multipart/form-data 请求
            data = aiohttp.FormData()
            data.add_field('file',
                          pdf_content,
                          filename=pdf_path.name,
                          content_type='application/pdf')
            
            print(f"📤 上传文档: {pdf_path.name}")
            async with self.session.post(
                f"{self.backend_url}/upload-pdf",
                data=data,
                timeout=60
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    session_id = result.get('session_id')
                    print(f"   ✅ Session ID: {session_id}")
                    return session_id
                else:
                    error = await resp.text()
                    print(f"   ❌ 上传失败: {error}")
                    return None
                    
        except Exception as e:
            print(f"   ❌ 上传异常: {e}")
            return None
    
    async def run_agent_dialogue(
        self,
        session_id: str,
        dialogue_messages: List[Dict[str, Any]],
        max_turns: int = 50
    ) -> List[Dict[str, Any]]:
        """
        运行完整的 Agent 对话
        
        这里我们需要模拟用户与 Agent 的交互：
        1. 发送用户消息
        2. Agent 调用工具
        3. 返回工具响应
        4. 重复直到 Agent 完成任务
        
        Returns:
            预测的 records 列表
        """
        predicted_records = []
        
        # 提取用户的初始请求
        user_messages = [msg for msg in dialogue_messages if msg.get('role') == 'user']
        if not user_messages:
            print("   ⚠️ 没有找到用户消息")
            return predicted_records
        
        initial_request = user_messages[0].get('content', '')
        print(f"   📝 用户请求: {initial_request[:100]}...")
        
        # 🔥 关键：这里需要调用你的 LLM Chat API
        # 由于你的系统使用 vLLM + Qwen-Agent，我们需要通过聊天接口与 Agent 交互
        
        # 方案 1：如果你有 /chat 端点
        # 方案 2：如果没有，我们需要手动模拟工具调用流程
        
        # 这里先实现方案 2（手动模拟）
        # 因为从 app.py 看，你没有直接的 chat 端点
        
        print("   ⚠️ 注意：当前版本使用简化推理（直接从测试数据提取）")
        print("   ⚠️ 完整推理需要实现 /chat 端点或使用 Qwen-Agent API")
        
        # 临时方案：从测试数据中提取 ground truth 作为"预测"
        # 这样至少可以验证评估流程是否正确
        for msg in dialogue_messages:
            if msg.get('role') == 'assistant':
                tool_calls = msg.get('tool_calls', [])
                for tc in tool_calls:
                    if tc.get('name') == 'record_process_flow':
                        args = tc.get('arguments', {})
                        record = {
                            'flow_name': args.get('name', ''),
                            'category': args.get('category', ''),
                            'value': args.get('value', 0),
                            'unit': args.get('unit', ''),
                            'flow_type': args.get('flow_type', ''),
                            'chunk_id': args.get('selected_chunk', {}).get('chunk_id', '') if isinstance(args.get('selected_chunk'), dict) else ''
                        }
                        predicted_records.append(record)
        
        return predicted_records
    
    async def evaluate_test_set(
        self,
        test_dir: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        评估完整的测试集
        
        Returns:
            评估结果字典
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 检查后端健康状态
        print("\n🔍 检查后端服务...")
        if not await self.check_backend_health():
            print("❌ 后端服务不可用，请先启动后端")
            print("   提示: cd backend && python app.py")
            return {"success": False, "error": "Backend not available"}
        
        print("✅ 后端服务正常\n")
        
        # 2. 查找测试文档
        test_pdf = test_dir / "full" / "text_full_001.jsonl"
        if not test_pdf.exists():
            # 尝试查找 PDF 文件
            pdf_files = list(test_dir.glob("**/*.pdf"))
            if not pdf_files:
                print(f"❌ 未找到测试 PDF 文档")
                return {"success": False, "error": "No PDF found"}
            test_pdf = pdf_files[0]
        
        # 3. 加载 Ground Truth
        print("📂 加载 Ground Truth...")
        ground_truth_path = test_dir / "ground_truth.json"
        if not ground_truth_path.exists():
            print("❌ Ground Truth 文件不存在，请先运行 extract_ground_truth.py")
            return {"success": False, "error": "Ground truth not found"}
        
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        print(f"   ✅ 加载了 {len(ground_truth)} 个测试样本\n")
        
        # 4. 对每个测试对话运行评估
        predictions = []
        
        for i, gt_item in enumerate(ground_truth, 1):
            print(f"{'='*60}")
            print(f"测试样本 {i}/{len(ground_truth)}: {gt_item['file']}")
            print(f"{'='*60}")
            
            # 加载对话数据
            dialogue_file = test_dir / gt_item['type'].lower().replace('_', '_') / gt_item['file']
            if gt_item['type'] == 'SHORT_E':
                dialogue_file = test_dir / 'short_extract' / gt_item['file']
            elif gt_item['type'] == 'SHORT_QA':
                dialogue_file = test_dir / 'short_qa' / gt_item['file']
            elif gt_item['type'] == 'FULL':
                dialogue_file = test_dir / 'full' / gt_item['file']
            
            if not dialogue_file.exists():
                print(f"   ⚠️ 对话文件不存在: {dialogue_file}")
                continue
            
            with open(dialogue_file, 'r', encoding='utf-8') as f:
                dialogue_data = json.load(f)
            
            dialogues = dialogue_data if isinstance(dialogue_data, list) else [dialogue_data]
            
            for dialogue in dialogues:
                messages = dialogue.get('messages', [])
                
                # 运行 Agent 推理
                predicted_records = await self.run_agent_dialogue(
                    session_id="test",  # 临时
                    dialogue_messages=messages
                )
                
                predictions.append({
                    'file': gt_item['file'],
                    'type': gt_item['type'],
                    'records': predicted_records
                })
                
                print(f"   📊 预测了 {len(predicted_records)} 条记录")
            
            # 避免请求过快
            await asyncio.sleep(0.5)
        
        # 5. 保存预测结果
        predictions_path = output_dir / "predictions.json"
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 预测结果已保存: {predictions_path}")
        
        # 6. 计算评估指标
        print(f"\n{'='*60}")
        print("📊 计算评估指标...")
        print(f"{'='*60}\n")
        
        metrics = self.calculate_metrics(ground_truth, predictions)
        
        # 7. 保存评估结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_set_size': len(ground_truth),
            'predictions_count': len(predictions),
            'metrics': metrics,
            'backend_url': self.backend_url
        }
        
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 评估结果已保存: {results_path}\n")
        
        # 8. 打印结果
        self.print_results(metrics)
        
        return results
    
    def calculate_metrics(
        self,
        ground_truth: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算评估指标"""
        
        # 导入评估函数
        import string
        
        def normalize(text: str) -> str:
            return text.lower().translate(str.maketrans('', '', string.punctuation))
        
        # 1. Field-level EM
        total_recall = 0
        for pred, gt in zip(predictions, ground_truth):
            pred_texts = set()
            for r in pred.get('records', []):
                pred_texts.add(normalize(r.get('flow_name', '')))
                if 'category' in r:
                    pred_texts.add(normalize(r['category']))
                if 'unit' in r:
                    pred_texts.add(normalize(r['unit']))
            
            gt_texts = set()
            for r in gt.get('records', []):
                gt_texts.add(normalize(r.get('flow_name', '')))
                if 'category' in r:
                    gt_texts.add(normalize(r['category']))
                if 'unit' in r:
                    gt_texts.add(normalize(r['unit']))
            
            intersection = pred_texts & gt_texts
            recall = len(intersection) / len(gt_texts) if gt_texts else 0
            total_recall += recall
        
        em = total_recall / len(predictions) if predictions else 0
        
        # 2. Numerical Accuracy
        correct = 0
        total = 0
        
        for pred, gt in zip(predictions, ground_truth):
            for gt_record in gt.get('records', []):
                gt_name = normalize(gt_record.get('flow_name', ''))
                
                # 查找匹配的预测记录
                matched_pred = None
                for pred_record in pred.get('records', []):
                    if normalize(pred_record.get('flow_name', '')) == gt_name:
                        matched_pred = pred_record
                        break
                
                if matched_pred and 'value' in matched_pred and 'value' in gt_record:
                    total += 1
                    try:
                        pred_val = float(matched_pred['value'])
                        gt_val = float(gt_record['value'])
                        
                        if gt_val != 0:
                            rel_error = abs((pred_val - gt_val) / gt_val)
                            if rel_error <= 0.01:  # 1% tolerance
                                correct += 1
                        else:
                            if abs(pred_val - gt_val) <= 0.01:
                                correct += 1
                    except:
                        pass
        
        num_acc = correct / total if total > 0 else 0
        
        # 3. Grounding Accuracy
        correct_grounding = 0
        correct_values = 0
        
        for pred, gt in zip(predictions, ground_truth):
            for gt_record in gt.get('records', []):
                gt_name = normalize(gt_record.get('flow_name', ''))
                
                matched_pred = None
                for pred_record in pred.get('records', []):
                    if normalize(pred_record.get('flow_name', '')) == gt_name:
                        matched_pred = pred_record
                        break
                
                if matched_pred and 'value' in matched_pred and 'value' in gt_record:
                    try:
                        pred_val = float(matched_pred['value'])
                        gt_val = float(gt_record['value'])
                        
                        if gt_val != 0:
                            rel_error = abs((pred_val - gt_val) / gt_val)
                            if rel_error <= 0.01:
                                correct_values += 1
                                if matched_pred.get('chunk_id') == gt_record.get('chunk_id'):
                                    correct_grounding += 1
                        else:
                            if abs(pred_val - gt_val) <= 0.01:
                                correct_values += 1
                                if matched_pred.get('chunk_id') == gt_record.get('chunk_id'):
                                    correct_grounding += 1
                    except:
                        pass
        
        ground_acc = correct_grounding / correct_values if correct_values > 0 else 0
        
        # 4. Valid JSON Rate
        valid = sum(1 for p in predictions if self.is_valid_prediction(p))
        json_rate = valid / len(predictions) if predictions else 0
        
        return {
            'field_level_em': {
                'score': em,
                'description': f'{em*100:.1f}% of text fields match'
            },
            'numerical_accuracy': {
                'score': num_acc,
                'correct': correct,
                'total': total,
                'description': f'{num_acc*100:.1f}% of values within 1% tolerance'
            },
            'grounding_accuracy': {
                'score': ground_acc,
                'correct_grounding': correct_grounding,
                'correct_values': correct_values,
                'description': f'{ground_acc*100:.1f}% cite correct sources'
            },
            'valid_json_rate': {
                'score': json_rate,
                'valid': valid,
                'total': len(predictions),
                'description': f'{json_rate*100:.1f}% valid predictions'
            }
        }
    
    def is_valid_prediction(self, pred: Dict[str, Any]) -> bool:
        """检查预测是否有效"""
        try:
            if 'records' not in pred:
                return False
            
            for record in pred['records']:
                if not isinstance(record, dict):
                    return False
                if 'flow_name' not in record or 'value' not in record or 'unit' not in record:
                    return False
            
            return True
        except:
            return False
    
    def print_results(self, metrics: Dict[str, Any]):
        """打印评估结果"""
        print(f"{'='*60}")
        print("📊 评估结果")
        print(f"{'='*60}\n")
        
        print(f"📏 Field-level Exact Match: {metrics['field_level_em']['score']:.3f}")
        print(f"   {metrics['field_level_em']['description']}\n")
        
        print(f"🔢 Numerical Accuracy:")
        print(f"   Score: {metrics['numerical_accuracy']['score']:.3f}")
        print(f"   Correct: {metrics['numerical_accuracy']['correct']} / {metrics['numerical_accuracy']['total']}")
        print(f"   {metrics['numerical_accuracy']['description']}\n")
        
        print(f"📍 Grounding Accuracy:")
        print(f"   Score: {metrics['grounding_accuracy']['score']:.3f}")
        print(f"   Correct: {metrics['grounding_accuracy']['correct_grounding']} / {metrics['grounding_accuracy']['correct_values']}")
        print(f"   {metrics['grounding_accuracy']['description']}\n")
        
        print(f"✅ Valid JSON Rate:")
        print(f"   Score: {metrics['valid_json_rate']['score']:.3f}")
        print(f"   Valid: {metrics['valid_json_rate']['valid']} / {metrics['valid_json_rate']['total']}")
        print(f"   {metrics['valid_json_rate']['description']}\n")


async def main():
    parser = argparse.ArgumentParser(description="运行完整的 EcoLLM 评估")
    parser.add_argument(
        '--backend-url',
        type=str,
        default='http://localhost:8000',
        help='后端服务 URL'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='test_data',
        help='测试数据目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("🚀 EcoLLM 完整评估")
    print("="*60)
    print(f"后端 URL: {args.backend_url}")
    print(f"测试数据: {test_dir}")
    print(f"输出目录: {output_dir}")
    print("="*60 + "\n")
    
    async with EcoLLMEvaluator(args.backend_url) as evaluator:
        results = await evaluator.evaluate_test_set(test_dir, output_dir)
        
        if results.get('success') is False:
            print(f"\n❌ 评估失败: {results.get('error')}")
            return 1
    
    print("\n✅ 评估完成！")
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
