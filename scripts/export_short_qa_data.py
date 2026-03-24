#!/usr/bin/env python3
"""
短对话 QA 场景导出脚本

功能：
1. 从 MongoDB 导出短对话数据（1-2 个数据提取）
2. 转换为 QA 场景：删除 record_* tool calls，直接返回答案
3. 保留 search_document 和 tool response
4. 生成自然的 user content（查询而非提取）

转换逻辑：
- MongoDB: search → record_process_flow(value=147.26, unit=kWh)
- 训练数据: search → "The process consumed 147.26 kWh of electricity."

作者：AI Assistant
版本：v1.0
日期：2025-11-29
"""

import json
import os
import sys
from typing import List, Dict, Optional
import argparse
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.mongodb_manager import mongodb_manager


class ShortQAExporter:
    """短对话 QA 场景导出器"""
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "lci_database"):
        """
        初始化导出器
        
        Args:
            mongo_uri: MongoDB连接URI
            db_name: 数据库名称
        """
        mongodb_manager.initialize(mongodb_uri=mongo_uri, db_name=db_name)
        self.db = mongodb_manager.get_database()
    
    def get_system_prompt_with_chunks(self, session_id: str) -> str:
        """
        获取带 chunk preview 的系统提示词
        
        Args:
            session_id: 会话ID
            
        Returns:
            系统提示词
        """
        # 基础系统提示词
        base_prompt = """You are an expert LCA assistant for Additive Manufacturing.

## Core Task
Answer questions about LCI data from documents.

## Tools
- search_document: Search text segments containing data via keywords

## Key Guidelines
1. **Search Strategy**: Combine related keywords (e.g., `["electricity", "power"]`) in one search
2. **Answer Format**: Provide clear, concise answers with specific values and units
3. **Missing Data**: If data is not found, clearly state this

**Remember**: Be helpful and accurate. Provide quantitative data when available."""
        
        # 获取 chunk preview（chunk 0 和 1）
        chunks = list(self.db.document_chunks.find({
            "session_id": session_id,
            "chunk_id": {"$in": ["0", "1"]}
        }).sort("chunk_id", 1))
        
        if not chunks:
            return base_prompt
        
        # 获取文档信息
        session = self.db.sessions.find_one({"session_id": session_id})
        doc_name = "Unknown"
        if session:
            doc_name = session.get("document_name", "Unknown")
        
        # 构建 context section
        context_section = f"""

**DOCUMENT CONTEXT**: 
A PDF document has been uploaded and is ready for analysis.
- Document Name: "{doc_name}"
- Document ID: {session_id[:8]}...
"""
        
        # 添加 chunk previews
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", "0")
            content = chunk.get("content", "")
            preview = content[:500] + "..." if len(content) > 500 else content
            
            if chunk_id == "0":
                context_section += f'\n**CHUNK 0 PREVIEW** (Executive Summary / Introduction):\n"{preview}"\n'
            else:
                context_section += f'\n**CHUNK 1 PREVIEW**:\n"{preview}"\n'
        
        # 添加 session injection 说明
        context_section += """
**AUTOMATIC SESSION INJECTION**: 
When you call search_document, the system will AUTOMATICALLY inject the session_id for you.

You do NOT need to ask the user for session_id. Just call the tool directly with the query parameter."""
        
        return base_prompt + context_section
    
    def build_search_tool_response(self, search_query, search_context: List[Dict]) -> str:
        """
        构建搜索工具响应
        
        Args:
            search_query: 搜索查询
            search_context: 搜索结果列表
            
        Returns:
            Tool response文本
        """
        results = []
        for ctx in search_context:
            results.append({
                "chunk_id": ctx.get("chunk_id"),
                "content": ctx.get("content")
            })
        
        response = {
            "success": True,
            "message": f"Found {len(results)} relevant results",
            "results": results
        }
        
        # 智能判断：批量查询 vs 单查询
        if isinstance(search_query, list) and len(search_query) > 1:
            response["queries"] = search_query
        else:
            response["query"] = search_query[0] if isinstance(search_query, list) else search_query
        
        return f"<tool_response>\n{json.dumps(response, ensure_ascii=False, indent=2)}\n</tool_response>"
    
    def _build_search_arguments(self, session_id: str, search_query) -> Dict:
        """
        构建 search_document 的 arguments
        
        Args:
            session_id: 会话ID
            search_query: 搜索查询
            
        Returns:
            arguments字典
        """
        arguments = {"session_id": session_id}
        
        if isinstance(search_query, list) and len(search_query) > 1:
            arguments["queries"] = search_query
        else:
            arguments["query"] = search_query[0] if isinstance(search_query, list) else search_query
        
        return arguments
    
    def build_answer_from_record(self, action: Dict, is_calculation_result: bool = False) -> str:
        """
        从 record 构建自然的答案
        
        Args:
            action: MongoDB 中的 action 记录
            is_calculation_result: 是否为计算结果
            
        Returns:
            自然的答案文本
        """
        record_type = action.get("record_type")
        
        if record_type == "flow":
            name = action.get("name", "")
            value = action.get("value", "")
            unit = action.get("unit", "")
            category = action.get("category", "")
            flow_type = action.get("flow_type", "")
            
            # 🔥 NEW: 如果是计算结果，使用不同的表述
            if is_calculation_result:
                return f"Based on the calculation, the {name.lower()} is {value} {unit}."
            
            # 生成自然的答案
            if flow_type == "Input":
                return f"The process consumed {value} {unit} of {name.lower()}."
            elif flow_type == "Output":
                if category == "Product":
                    return f"The process produced {value} {unit} of {name.lower()}."
                elif category == "Waste":
                    return f"The process generated {value} {unit} of {name.lower()} waste."
                elif category == "Emission":
                    return f"The process emitted {value} {unit} of {name.lower()}."
                else:
                    return f"The output is {value} {unit} of {name.lower()}."
            else:
                return f"The {name.lower()} is {value} {unit}."
        
        elif record_type == "scope":
            param_name = action.get("parameter_name", "")
            description = action.get("description", "")
            return f"The functional unit is: {description if description else param_name}"
        
        elif record_type == "parameter":
            # 🔥 NEW: 支持 parameter（用于 calculation）
            param_name = action.get("parameter_name", "")
            value = action.get("value", "")
            unit = action.get("unit", "")
            return f"The {param_name.lower()} is {value} {unit}."
        
        else:
            return "I found the requested information in the document."
    
    def export_session_to_qa_format(self, session_id: str) -> List[Dict]:
        """
        导出单个 session 为 QA 格式
        
        转换规则：
        1. 保留所有 search_document 和对应的 tool response
        2. 删除所有 record_* tool calls
        3. 将 record_* 的数据转换为 assistant 的直接回答
        4. User content 留空（后续用 CAMEL AI 生成）
        5. 🔥 NEW: 支持 calculation、pivot、smart_skip 场景
        
        Args:
            session_id: 会话ID
            
        Returns:
            样本列表
        """
        # 查询该 session 的所有 actions
        actions = list(self.db.lca_actions.find({
            "session_id": session_id
        }).sort("action_id", 1))
        
        if not actions:
            logger.warning(f"Session {session_id} 没有找到任何记录")
            return []
        
        logger.info(f"找到 {len(actions)} 个 action")
        
        # 过滤掉 document_preview
        actions = [a for a in actions if a.get("record_type") != "document_preview"]
        logger.info(f"过滤后剩余 {len(actions)} 个 action")
        
        # 创建消息列表
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt_with_chunks(session_id)
            },
            {
                "role": "user",
                "content": ""  # 待 CAMEL AI 生成
            }
        ]
        
        # 跟踪上一次的 search_query
        last_search_query = None
        
        # 收集所有的 record 数据（用于生成最终答案）
        recorded_data = []
        
        # 🔥 NEW: 跟踪 calculation 链（parameter → calculation → flow）
        calculation_chain = {}  # {action_id: calculation_result}
        
        for action in actions:
            record_type = action.get("record_type")
            intent = action.get("intent", "")
            action_id = action.get("action_id")
            
            # 跳过 document_preview
            if record_type == "document_preview":
                continue
            
            # 🔥 NEW: 处理 pivot（失败搜索）
            if record_type == "pivot" or intent == "pivot_query":
                failed_query = action.get("failed_query")
                failed_context = action.get("failed_context", [])
                
                if failed_query and failed_query != last_search_query:
                    # 添加失败的搜索
                    messages.append({
                        "role": "assistant",
                        "reasoning_content": "",
                        "tool_calls": [{
                            "name": "search_document",
                            "arguments": self._build_search_arguments(session_id, failed_query)
                        }]
                    })
                    
                    search_response = self.build_search_tool_response(failed_query, failed_context)
                    messages.append({
                        "role": "tool",
                        "content": search_response
                    })
                    
                    last_search_query = failed_query
                continue
            
            # 🔥 NEW: 处理 smart_skip
            if intent == "smart_skip":
                smart_skip_query = action.get("search_query")
                skipped_chunk = action.get("skipped_chunk", {})
                
                if smart_skip_query and smart_skip_query != last_search_query:
                    messages.append({
                        "role": "assistant",
                        "reasoning_content": "",
                        "tool_calls": [{
                            "name": "search_document",
                            "arguments": self._build_search_arguments(session_id, smart_skip_query)
                        }]
                    })
                    
                    search_context = [skipped_chunk] if skipped_chunk else []
                    search_response = self.build_search_tool_response(smart_skip_query, search_context)
                    messages.append({
                        "role": "tool",
                        "content": search_response
                    })
                    
                    last_search_query = smart_skip_query
                continue
            
            # 处理 search
            search_query = action.get("search_query")
            search_context = action.get("search_context", [])
            
            if search_query and search_query != last_search_query:
                # 添加 search_document 的 assistant 消息
                messages.append({
                    "role": "assistant",
                    "reasoning_content": "",  # 待 CAMEL AI 生成
                    "tool_calls": [{
                        "name": "search_document",
                        "arguments": self._build_search_arguments(session_id, search_query)
                    }]
                })
                
                # 添加 search_document 的 tool 响应
                search_response = self.build_search_tool_response(search_query, search_context)
                messages.append({
                    "role": "tool",
                    "content": search_response
                })
                
                last_search_query = search_query
            
            # 🔥 NEW: 处理 calculation
            if record_type == "calculation":
                calc_result = action.get("calculation_result")
                calc_unit = action.get("calculation_unit", "")
                calculation_chain[action_id] = {"result": calc_result, "unit": calc_unit}
                continue
            
            # 收集 record 数据（不添加 tool call）
            if record_type in ["flow", "scope", "parameter"]:
                # 🔥 NEW: 检查是否链接到 calculation
                link_to = action.get("link_to")
                is_calc_result = link_to in calculation_chain
                
                recorded_data.append({
                    "action": action,
                    "is_calculation_result": is_calc_result
                })
        
        # 生成最终的答案（基于所有 recorded_data）
        if recorded_data:
            # 如果只有一个数据，直接回答
            if len(recorded_data) == 1:
                data_item = recorded_data[0]
                answer = self.build_answer_from_record(
                    data_item["action"],
                    is_calculation_result=data_item["is_calculation_result"]
                )
            else:
                # 如果有多个数据，组合回答
                answers = []
                for data_item in recorded_data:
                    answers.append(self.build_answer_from_record(
                        data_item["action"],
                        is_calculation_result=data_item["is_calculation_result"]
                    ))
                answer = " ".join(answers)
            
            # 添加最终的 assistant 回复
            messages.append({
                "role": "assistant",
                "reasoning_content": "",  # 待 CAMEL AI 生成
                "content": answer
            })
        
        return [{
            "messages": messages
        }]
    
    def export_sessions(self, session_ids: List[str], output_file: str):
        """
        批量导出 sessions
        
        Args:
            session_ids: Session ID 列表
            output_file: 输出文件路径
        """
        all_samples = []
        
        for session_id in session_ids:
            logger.info(f"\n处理 session: {session_id}")
            samples = self.export_session_to_qa_format(session_id)
            all_samples.extend(samples)
        
        # 保存结果
        logger.info(f"\n保存 {len(all_samples)} 个样本到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ 导出完成！")


def main():
    parser = argparse.ArgumentParser(description="短对话 QA 场景导出器")
    parser.add_argument("--session-ids", required=True, help="Session IDs（逗号分隔）")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/", help="MongoDB URI")
    parser.add_argument("--db-name", default="lci_database", help="数据库名称")
    
    args = parser.parse_args()
    
    # 解析 session IDs
    session_ids = [s.strip() for s in args.session_ids.split(",")]
    
    # 创建导出器
    exporter = ShortQAExporter(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name
    )
    
    # 导出数据
    exporter.export_sessions(session_ids, args.output)


if __name__ == "__main__":
    main()
