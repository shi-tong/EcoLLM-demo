"""
训练数据导出脚本 - Chat Template格式

功能：
1. 从 MongoDB 导出专家工作台标注的数据
2. 转换为 Chat Template 格式（用于 LLM 训练）
3. 生成 JSONL 文件供 CAMEL AI 补充 <think> 内容
4. 确保与实际 API 调用和响应格式完全一致

作者：AI Assistant
日期：2025-11-16
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


class TrainingDataExporter:
    """训练数据导出器 - Chat Template格式"""
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "lci_database"):
        """
        初始化导出器
        
        Args:
            mongo_uri: MongoDB连接URI
            db_name: 数据库名称
        """
        mongodb_manager.initialize(mongodb_uri=mongo_uri, db_name=db_name)
        self.db = mongodb_manager.get_database()
        
    def get_system_prompt(self) -> str:
        """
        获取系统提示词 (v5.4.1 - 整合实际服务经验)
        
        遵循 "Less is More" 原则：极简、聚焦、指令式
        整合实际 LLM 服务的关键规则（Session ID 自动注入）
        """
        return """You are an expert LCA assistant for Additive Manufacturing.

## Core Task
Extract quantitative LCI data from documents or answer specific questions.

## Tools
- search_document: Search text segments containing data via keywords
- define_lca_scope: Record Functional Unit
- record_process_flow: Record LCI flows (quantitative values)
- record_parameter: Record intermediate parameters for calculation
- execute_calculation: Calculate derived values
- get_session_summary: Check recorded data

## Strategic Workflow
1. **Anchor (Functional Unit)**: Identify the study basis (e.g., "manufacturing a 316L impeller by SLM"). Record via `define_lca_scope`.
2. **Inputs**: Extract Material, Energy, Gas, Cooling. Use `record_process_flow` or `record_parameter`.
3. **Outputs**: Extract Waste, Emissions. **Crucial**: Record Product quantitative flow here via `record_process_flow`.
4. **Validation** (Optional): Check completeness for full inventory tasks.
*For short QA: Jump directly to the requested info.*

## LCI Categories (11 types)
**Input**: Raw Material, Process Energy, Post-processing Energy, Feedstock Energy, Gas, Cooling Media
**Output**: Product, Recovered Material, Waste, Emission
**Scope**: Functional Unit

## Key Guidelines
1. **Batch Search**: Always combine related keywords (e.g., `["electricity", "power"]`) in one search.
2. **Calc vs Record**: 
   - Explicit data (e.g., "120 kWh") → `record_process_flow`
   - Needs calculation (e.g., "500W, 2h") → `record_parameter` + `execute_calculation` → `record_process_flow`
3. **Energy Classification**:
   - *Process*: Printing/Machine operation
   - *Post-processing*: Heat treatment/Machining
   - *Feedstock*: Powder production
4. **Note Field**: Use for context (e.g., "SLM machine", "Atomization") to distinguish same-name flows or key qualifiers (e.g., "99.9% purity", "recycled") to enrich flows.
5. **Session ID**: Include in tool calls for traceability. The system will handle it automatically.
6. **Summary Check**: Use `get_session_summary` to review recorded data when needed.
7. **Missing Data**: Skip if not found. Do not hallucinate.

**Remember**: Be concise. Extract quantitative data accurately."""
    
    def get_system_prompt_with_chunks(self, session_id: str) -> str:
        """
        获取带 chunk preview 的系统提示词（用于训练数据）
        
        与推理时的 system prompt 保持一致（local_qwen_service.py）
        
        优先从 document_preview 记录中读取，否则从 search_context 中提取
        
        Args:
            session_id: 会话ID
            
        Returns:
            带 chunk preview 的 system prompt
        """
        base_prompt = self.get_system_prompt()
        
        # 🔥 从 document_preview 记录中读取（用户在工作台点击"Record Preview"按钮）
        preview_record = self.db.lca_actions.find_one({
            "session_id": session_id,
            "record_type": "document_preview"
        })
        
        if not preview_record:
            # 如果没有 document_preview 记录，返回基础 prompt
            logger.warning(f"⚠️ 未找到 document_preview 记录: {session_id}. 请在工作台点击 'Record Preview' 按钮。")
            return base_prompt
        
        # 从 document_preview 记录中读取
        doc_preview = preview_record.get("document_preview", {})
        
        chunk_0_content = None
        chunk_1_content = None
        
        chunk_0_data = doc_preview.get("chunk_0")
        if chunk_0_data:
            chunk_0_content = chunk_0_data.get("content")
        
        chunk_1_data = doc_preview.get("chunk_1")
        if chunk_1_data:
            chunk_1_content = chunk_1_data.get("content")
        
        document_name = preview_record.get("metadata", {}).get("document_name", "Unknown Document")
        
        logger.info(f"✅ 从 document_preview 记录中读取 chunks: {session_id}")
        
        # 构建文档上下文（与 local_qwen_service.py 保持一致）
        context_section = f"""

**DOCUMENT CONTEXT**: 
A PDF document has been uploaded and is ready for analysis.
- Document Name: "{document_name}"
- Document ID: {session_id[:8]}...
"""
        
        # 添加 chunk previews（完整内容，不添加 "..."）
        if chunk_0_content:
            context_section += f"""
**CHUNK 0 PREVIEW** (Executive Summary / Introduction):
"{chunk_0_content}"
"""
        
        if chunk_1_content:
            context_section += f"""
**CHUNK 1 PREVIEW**:
"{chunk_1_content}"
"""
        
        context_section += """
**AUTOMATIC SESSION INJECTION**: 
When you call document-related tools (search_document, record_parameter, record_calculation, record_process_flow, get_session_summary, define_lca_scope, record_pivot_failure), the system will AUTOMATICALLY inject the session_id for you.

You do NOT need to ask the user for session_id. Just call the tools directly with other required parameters (e.g., "query" for search_document)."""
        
        return base_prompt + context_section
    
    def _infer_query_from_category(self, category: str) -> Optional[str]:
        """
        根据类别推断搜索查询
        
        Args:
            category: 类别名称（如 "Energy", "Ancillary Material"）
            
        Returns:
            推断的搜索查询，如果无法推断则返回None
        """
        category_query_map = {
            # Input 类别
            "Raw Material": "powder material feedstock",
            "Process Energy": "energy electricity power consumption",
            "Post-processing Energy": "heat treatment machining post-processing energy",
            "Feedstock Energy": "powder production atomization feedstock energy",
            "Gas": "gas argon nitrogen shielding",
            "Cooling Media": "water cooling liquid coolant",
            # Output 类别
            "Product": "product part manufactured output",
            "Recovered Material": "recovered reuse recycled powder",
            "Waste": "waste scrap trimmings dust disposal",
            "Emission": "emissions CO2 exhaust gas",
            # 兼容旧名称
            "Energy": "energy electricity power consumption",
            "Ancillary Material": "gas argon nitrogen auxiliary",
            "Water": "water cooling liquid",
            "By-product": "recovered reuse recycled powder",
            "Emissions": "emissions CO2 exhaust gas",
            "Wastewater": "wastewater effluent discharge",
            "Transport Service": "transport shipping logistics"
        }
        return category_query_map.get(category)
    
    def get_session_summary_text(self, session_id: str, before_action_id: str = None) -> str:
        """
        获取会话摘要文本（LLM视图）
        
        🔥 重要：只包含flow和parameter，不包含calculation，保证训练-推理一致性
        
        Args:
            session_id: 会话ID
            before_action_id: 只包含此action_id之前的记录
            
        Returns:
            会话摘要文本（LLM视图）
        """
        # 查询该session的所有之前的记录
        query = {"session_id": session_id}
        if before_action_id:
            # 只获取之前的记录
            query["action_id"] = {"$lt": before_action_id}
        
        actions = list(self.db.lca_actions.find(query).sort("action_id", 1))
        
        if not actions:
            return "No previous records in this session."
        
        # 🔥 LLM视图：统计scope, flow和parameter
        scopes = [a for a in actions if a.get("record_type") == "scope"]
        flows = [a for a in actions if a.get("record_type") == "flow"]
        parameters = [a for a in actions if a.get("record_type") == "parameter"]
        
        summary_parts = []
        summary_parts.append(f"Total Actions: {len(scopes) + len(flows) + len(parameters)}")
        summary_parts.append(f"Scopes: {len(scopes)}, Flows: {len(flows)}, Parameters: {len(parameters)}")
        summary_parts.append("")
        
        if scopes:
            summary_parts.append("LCA Scope:")
            for scope in scopes:
                param_name = scope.get('parameter_name', 'Unknown')
                description = scope.get('description', '')
                summary_parts.append(f"  - {param_name}: {description}")
            summary_parts.append("")
        
        if flows:
            summary_parts.append("Recorded Flows:")
            for flow in flows:
                summary_parts.append(
                    f"  - {flow.get('name')}: {flow.get('value')} {flow.get('unit')} "
                    f"({flow.get('flow_type')}/{flow.get('category')})"
                )
            summary_parts.append("")
        
        if parameters:
            summary_parts.append("Recorded Parameters:")
            for param in parameters:
                summary_parts.append(
                    f"  - {param.get('parameter_name')}: "
                    f"{param.get('parameter_value')} {param.get('parameter_unit', '')}"
                )
            summary_parts.append("")
        
        # 🔥 不包含calculation统计
        
        return "\n".join(summary_parts)
    
    def build_tool_call(self, action: Dict) -> Dict:
        """
        构建Tool Call（Assistant消息中的工具调用）
        
        Args:
            action: MongoDB中的action记录
            
        Returns:
            Tool call字典
        """
        record_type = action.get("record_type")
        session_id = action.get("session_id")
        
        if record_type == "parameter":
            # 清理selected_chunk，移除内部字段
            selected_chunk = action.get("selected_chunk", {})
            clean_chunk = {
                "chunk_id": selected_chunk.get("chunk_id"),
                "content": selected_chunk.get("content")
            }
            
            # 构建基本参数
            arguments = {
                "session_id": session_id,
                "parameter_name": action.get("parameter_name"),
                "parameter_value": action.get("parameter_value"),
                "parameter_unit": action.get("parameter_unit"),
                "selected_chunk": clean_chunk
            }
            
            # 添加可选字段（如果存在）
            if action.get("note"):
                arguments["note"] = action.get("note")
            
            return {
                "name": "record_parameter",
                "arguments": arguments
            }
        
        elif record_type == "flow":
            # 清理selected_chunk，移除内部字段
            selected_chunk = action.get("selected_chunk", {})
            clean_chunk = {
                "chunk_id": selected_chunk.get("chunk_id"),
                "content": selected_chunk.get("content")
            }
            
            # 构建基本参数
            arguments = {
                "session_id": session_id,
                "flow_type": action.get("flow_type"),
                "category": action.get("category"),
                "name": action.get("name"),
                "value": action.get("value"),
                "unit": action.get("unit"),
                "selected_chunk": clean_chunk
            }
            
            # 添加可选字段（如果存在）
            if action.get("note"):
                arguments["note"] = action.get("note")
            if action.get("process_name"):
                arguments["process_name"] = action.get("process_name")
            if action.get("cas_number"):
                arguments["cas_number"] = action.get("cas_number")
            if action.get("location"):
                arguments["location"] = action.get("location")
            # 注意：link_to 是内部字段，用于工作台标注时建立动作链条
            # 不应导出到训练数据中，LLM 推理时不会用到
            
            return {
                "name": "record_process_flow",
                "arguments": arguments
            }
        
        elif record_type == "calculation":
            return {
                "name": "execute_calculation",
                "arguments": {
                    "session_id": session_id,
                    "expression": action.get("calculation_expression"),
                    "dependencies": action.get("data_dependencies", [])
                }
            }
        
        elif record_type == "scope":
            # 清理selected_chunk，移除内部字段
            selected_chunk = action.get("selected_chunk", {})
            clean_chunk = {
                "chunk_id": selected_chunk.get("chunk_id"),
                "content": selected_chunk.get("content")
            }
            
            # 构建基本参数
            arguments = {
                "session_id": session_id,
                "parameter_name": action.get("parameter_name"),
                "description": action.get("description"),
                "value": action.get("value"),
                "unit": action.get("unit"),
                "source_content": action.get("source_content"),
                "selected_chunk": clean_chunk
            }
            
            # 添加可选字段（如果存在）
            if action.get("note"):
                arguments["note"] = action.get("note")
            
            return {
                "name": "define_lca_scope",
                "arguments": arguments
            }
        
        elif record_type == "summary" or record_type == "summary_check" or action.get("intent") == "get_summary":
            # 支持get_session_summary查询工具
            return {
                "name": "get_session_summary",
                "arguments": {
                    "session_id": session_id,
                    "format": "text"  # LLM使用text格式
                }
            }
        
        else:
            raise ValueError(f"Unknown record_type: {record_type}")
    
    def build_tool_response(self, action: Dict) -> str:
        """
        构建Tool Response（Tool消息内容）
        
        根据实际后端返回格式构建响应
        
        Args:
            action: MongoDB中的action记录
            
        Returns:
            Tool response文本（包含<tool_response>标签）
        """
        record_type = action.get("record_type")
        action_id = action.get("action_id")
        session_id = action.get("session_id")
        
        # 根据record_type构建不同的响应消息
        if record_type == "parameter":
            response = {
                "success": True,
                "message": f"Parameter recorded with action_id: {action_id}",
                "new_action_id": action_id
            }
        
        elif record_type == "flow":
            name = action.get("name", "")
            response = {
                "success": True,
                "message": f"Successfully recorded process flow: {name}",
                "data": {
                    "record_id": str(action.get("_id")),
                    "action_id": action_id,
                    "session_id": session_id,
                    "flow_type": action.get("flow_type"),
                    "category": action.get("category"),
                    "name": name,
                    "value": action.get("value"),
                    "unit": action.get("unit")
                }
            }
        
        elif record_type == "calculation":
            calc_result = action.get("calculation_result")
            calc_unit = action.get("calculation_unit", "")
            response = {
                "success": True,
                "message": "Calculation executed successfully",
                "result": calc_result,
                "unit": calc_unit
            }
        
        elif record_type == "scope":
            param_name = action.get("parameter_name", "")
            response = {
                "success": True,
                "message": f"Successfully defined LCA scope: {param_name}",
                "data": {
                    "record_id": str(action.get("_id")),
                    "action_id": action_id,
                    "session_id": session_id,
                    "parameter_name": param_name
                }
            }
        
        elif record_type == "summary" or record_type == "summary_check" or action.get("intent") == "get_summary":
            # 🔥 处理session summary响应（LLM视图：不包含calculation）
            # 直接使用实时查询，确保数据准确性（不依赖可能过时的snapshot）
            summary_text = self.get_session_summary_text(session_id, before_action_id=action_id)
            
            response = {
                "success": True,
                "summary": summary_text
            }
        
        else:
            response = {
                "success": True,
                "message": f"Successfully recorded {record_type}",
                "action_id": action_id
            }
        
        # 包装在<tool_response>标签中
        return f"<tool_response>\n{json.dumps(response, ensure_ascii=False, indent=2)}\n</tool_response>"
    
    def _build_search_arguments(self, session_id: str, search_query) -> Dict:
        """
        构建search_document的arguments（智能判断query vs queries）
        
        Args:
            session_id: 会话ID
            search_query: 搜索查询（字符串或列表）
            
        Returns:
            arguments字典
        """
        arguments = {"session_id": session_id}
        
        if isinstance(search_query, list) and len(search_query) > 1:
            # 批量查询：使用 queries 参数
            arguments["queries"] = search_query
        else:
            # 单查询：使用 query 参数
            arguments["query"] = search_query[0] if isinstance(search_query, list) else search_query
        
        return arguments
    
    def build_search_tool_response(self, search_query, search_context: List[Dict]) -> str:
        """
        构建搜索工具响应（与真实API格式一致）
        
        Args:
            search_query: 搜索查询（可能是字符串或列表）
            search_context: 搜索结果列表
            
        Returns:
            Tool response文本（包含<tool_response>标签）
        """
        
        # 格式化搜索结果（仅包含chunk_id和content，与真实API一致）
        results = []
        for ctx in search_context:
            results.append({
                "chunk_id": ctx.get("chunk_id"),
                "content": ctx.get("content")
            })
        
        # 与真实API格式一致（英文消息）
        response = {
            "success": True,
            "message": f"Found {len(results)} relevant results",
            "results": results
        }
        
        # 智能判断：批量查询 vs 单查询
        if isinstance(search_query, list) and len(search_query) > 1:
            # 批量查询：使用 queries 字段
            response["queries"] = search_query
        else:
            # 单查询：使用 query 字段
            response["query"] = search_query[0] if isinstance(search_query, list) else search_query
        
        return f"<tool_response>\n{json.dumps(response, ensure_ascii=False, indent=2)}\n</tool_response>"
    
    def export_session_to_chat_format(self, session_id: str) -> List[Dict]:
        """
        导出单个session为chat_template格式（完整多轮对话）
        
        ✅ 改进设计：每个session生成一个样本（而非每个action一个样本）
        
        消息结构：
        1. System: 系统提示词
        2. User: 用户问题（单个，描述整个session目标）
        3. Assistant: 第一个工具调用
        4. Tool: 第一个工具响应
        5. Assistant: 第二个工具调用
        6. Tool: 第二个工具响应
        ... (重复3-4步骤）
        
        Args:
            session_id: 会话ID
            
        Returns:
            样本列表（通常只有一个样本）
        """
        # 查询该session的所有actions
        # ✅ 按action_id排序确保顺序正确（ACT_0001, ACT_0002, ...）
        actions = list(self.db.lca_actions.find({
            "session_id": session_id
        }).sort("action_id", 1))
        
        if not actions:
            print(f"⚠️  Session {session_id} 没有找到任何记录")
            return []
        
        print(f"📝 找到 {len(actions)} 个action")
        
        # 🔥 过滤掉 document_preview（这是手动记录的，不是LLM调用的）
        actions = [a for a in actions if a.get("record_type") != "document_preview"]
        logger.info(f"过滤后剩余 {len(actions)} 个action（已排除 document_preview）")
        
        # 创建单个样本（包含整个session）
        # 🔥 使用带 chunk preview 的 system prompt（与推理时一致）
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt_with_chunks(session_id)
            },
            {
                "role": "user",
                "content": ""  # 待用户填写
            }
        ]
        
        # 按顺序添加每个action的messages
        # 跟踪上一次的search_query，避免重复插入search_document
        last_search_query = None
        
        for action in actions:
            try:
                record_type = action.get("record_type")
                intent = action.get("intent")
                
                # 🔥 过滤：跳过 document_preview 记录（不导出到训练数据）
                if record_type == "document_preview":
                    continue
                
                # 🔥 特殊处理：Pivot Query（失败路径）
                if record_type == "pivot" or intent == "pivot_query":
                    # Pivot只插入search_document及其响应，不插入record_pivot_query
                    failed_query = action.get("failed_query")
                    failed_context = action.get("failed_context", [])
                    
                    if failed_query and failed_query != last_search_query:
                        # 添加search_document的assistant消息
                        messages.append({
                            "role": "assistant",
                            "reasoning_content": "",  # CAMEL AI会生成：为什么搜索这个关键词
                            "tool_calls": [{
                                "name": "search_document",
                                "arguments": self._build_search_arguments(action.get("session_id"), failed_query)
                            }]
                        })
                        
                        # 添加search_document的tool响应（使用failed_context）
                        search_response = self.build_search_tool_response(failed_query, failed_context)
                        messages.append({
                            "role": "tool",
                            "content": search_response
                        })
                        
                        # 更新last_search_query
                        last_search_query = failed_query
                    
                    # ⚠️ 不添加record_pivot_query的tool call！
                    # Pivot记录只用于标记失败，LLM会直接进行下一次搜索
                    continue
                
                # 🔥 特殊处理：Smart Skip（智能跳过）
                if intent == "smart_skip":
                    # Smart skip只插入search_document及其响应，不插入record_smart_skip
                    # 因为LLM应该通过session summary判断数据已记录，然后直接进入下一个大类
                    skipped_chunk = action.get("skipped_chunk", {})
                    category = action.get("category", "Unknown")
                    
                    # 🔥 FIX: 使用MongoDB中的search_query（保留完整的多词关键词）
                    smart_skip_query = action.get("search_query")
                    
                    # 如果有search_query，说明进行了搜索但发现数据已记录
                    if smart_skip_query and smart_skip_query != last_search_query:
                            # 🔥 修复：先检查是否有上一次搜索的assistant消息（避免重复插入）
                            # 如果上一条消息不是search_document的assistant，则需要插入
                            need_search_call = True
                            if messages and messages[-1].get("role") == "assistant":
                                last_tool_calls = messages[-1].get("tool_calls", [])
                                # 🔥 FIX: 检查 tool_calls 是否非空且第一个是 search_document
                                if last_tool_calls and len(last_tool_calls) > 0 and last_tool_calls[0].get("name") == "search_document":
                                    # 上一条已经是search，直接复用
                                    need_search_call = False
                                # 🔥 FIX: 如果上一条是 smart skip（tool_calls 为空），需要合并
                                elif not last_tool_calls or len(last_tool_calls) == 0:
                                    # 上一条是 smart skip placeholder，将搜索合并到它的 tool_calls 中
                                    messages[-1]["tool_calls"] = [{
                                        "name": "search_document",
                                        "arguments": self._build_search_arguments(action.get("session_id"), smart_skip_query)
                                    }]
                                    need_search_call = False
                            
                            if need_search_call:
                                # 添加search_document的assistant消息（正常搜索，不带placeholder）
                                messages.append({
                                    "role": "assistant",
                                    "reasoning_content": "",  # CAMEL AI会生成正常的搜索reasoning
                                    "tool_calls": [{
                                        "name": "search_document",
                                        "arguments": self._build_search_arguments(action.get("session_id"), smart_skip_query)
                                    }]
                                })
                            
                            # 🔥 修复：使用保存的完整 search_context
                            # 从 action 中读取 search_context（如果有的话）
                            search_context = action.get("search_context", [])
                            
                            # 如果没有 search_context，回退到 skipped_chunk（兼容旧数据）
                            if not search_context and skipped_chunk:
                                search_context = [skipped_chunk]
                            
                            # 添加search_document的tool响应
                            search_response = self.build_search_tool_response(smart_skip_query, search_context)
                            messages.append({
                                "role": "tool",
                                "content": search_response
                            })
                            
                            # 🔥 添加assistant的skip reasoning（在看到tool response之后）
                            skip_reason = action.get("skip_reason", "already_recorded")
                            messages.append({
                                "role": "assistant",
                                "reasoning_content": f"[SMART_SKIP_PLACEHOLDER: {category} - {skip_reason}]",
                                "tool_calls": []  # 空的tool_calls表示不执行任何操作，直接跳过
                            })
                            
                            # 更新last_search_query
                            last_search_query = smart_skip_query
                    
                    # ⚠️ 不添加record_smart_skip的tool call！
                    # Smart skip记录只用于标记跳过，LLM会通过reasoning表达"数据已记录，跳过"
                    continue
                
                # 🔥 FIX: 在处理正常 action 之前，检查上一条消息是否是 Smart Skip
                # 如果是，需要将当前 action 的 tool_call 合并到它的 tool_calls 中
                should_merge_with_prev = False
                if messages and messages[-1].get("role") == "assistant":
                    last_tool_calls = messages[-1].get("tool_calls", [])
                    last_reasoning = messages[-1].get("reasoning_content", "")
                    # 如果上一条是 Smart Skip（tool_calls 为空且有 PLACEHOLDER）
                    if (not last_tool_calls or len(last_tool_calls) == 0) and "SMART_SKIP_PLACEHOLDER" in last_reasoning:
                        # 将当前 action 的 tool_call 添加到上一条消息
                        # 这样就避免了连续的 assistant 消息
                        should_merge_with_prev = True
                
                # ✅ 正常流程：检查是否需要先插入search_document调用
                current_search_query = action.get("search_query")
                search_context = action.get("search_context", [])
                
                # 如果有新的search_query（且不同于上一次），先插入search_document
                if current_search_query and current_search_query != last_search_query and search_context:
                    # 🔥 FIX: 如果需要合并到上一条 Smart Skip 消息
                    if should_merge_with_prev:
                        # 检查上一条消息是否是 assistant（安全检查）
                        if messages and messages[-1].get("role") == "assistant":
                            # 将 search_document 添加到上一条 assistant 消息的 tool_calls
                            messages[-1]["tool_calls"] = [{
                                "name": "search_document",
                                "arguments": self._build_search_arguments(action.get("session_id"), current_search_query)
                            }]
                        else:
                            # 上一条是 tool 或其他，插入新的 assistant 消息
                            messages.append({
                                "role": "assistant",
                                "reasoning_content": "",
                                "tool_calls": [{
                                    "name": "search_document",
                                    "arguments": self._build_search_arguments(action.get("session_id"), current_search_query)
                                }]
                            })
                    else:
                        # 添加search_document的assistant消息
                        messages.append({
                            "role": "assistant",
                            "reasoning_content": "",
                            "tool_calls": [{
                                "name": "search_document",
                                "arguments": self._build_search_arguments(action.get("session_id"), current_search_query)
                            }]
                        })
                    
                    # 添加search_document的tool响应
                    search_response = self.build_search_tool_response(current_search_query, search_context)
                    messages.append({
                        "role": "tool",
                        "content": search_response
                    })
                    
                    # 更新last_search_query
                    last_search_query = current_search_query
                
                # 构建工具调用（record_* 或 get_session_summary）
                tool_call = self.build_tool_call(action)
                
                # 🔥 FIX: 如果需要合并，将 tool_call 添加到上一条消息
                if should_merge_with_prev:
                    # 检查上一条消息是否是 assistant（安全检查，防止添加到 tool 消息）
                    if messages and messages[-1].get("role") == "assistant":
                        messages[-1]["tool_calls"] = [tool_call]
                    else:
                        # 上一条是 tool 或其他，插入新的 assistant 消息
                        messages.append({
                            "role": "assistant",
                            "reasoning_content": "",
                            "tool_calls": [tool_call]
                        })
                else:
                    # 添加assistant消息
                    messages.append({
                        "role": "assistant",
                        "reasoning_content": "",
                        "tool_calls": [tool_call]
                    })
                
                # 构建工具响应
                tool_response = self.build_tool_response(action)
                
                # 添加tool消息
                messages.append({
                    "role": "tool",
                    "content": tool_response
                })
                
            except Exception as e:
                print(f"⚠️  处理 {action.get('action_id')} 时出错: {e}")
                continue
        
        # 🔥 NEW: 添加最终的assistant回复（闭环）
        # 根据最后一个action类型决定回复风格
        if actions:
            last_action = actions[-1]
            last_record_type = last_action.get("record_type")
            
            logger.info(f"最后一个action: record_type={last_record_type}, tool_name={last_action.get('tool_name')}")
            
            # 场景1：完整提取 - 最后是summary
            if last_record_type == "summary_check" or last_action.get("tool_name") == "get_session_summary":
                logger.info("✅ 添加最终assistant回复（完整提取场景）")
                # 🔥 CAMEL AI会根据session summary生成完整的报告
                messages.append({
                    "role": "assistant",
                    "reasoning_content": "",  # CAMEL AI生成：完整流程的闭环总结
                    "content": ""  # CAMEL AI生成：结构化的LCI数据提取报告
                })
            
            # 场景2：简单任务 - 最后是record（但没有summary）
            # 这种情况可能是：用户只要求提取1-2个数据点
            # 注意：calculation 后通常会跟 flow，所以这里包含 calculation 也没问题
            elif last_record_type in ["flow", "parameter", "scope", "calculation"] and len(actions) <= 5:
                logger.info("✅ 添加最终assistant回复（简单任务场景）")
                # 🔥 CAMEL AI会生成简洁的任务完成确认
                messages.append({
                    "role": "assistant",
                    "reasoning_content": "",  # CAMEL AI生成：简单的任务完成确认
                    "content": ""  # CAMEL AI生成：简洁的回复
                })
        
        # 创建单个样本（不包含metadata）
        sample = {
            "messages": messages
        }
        
        return [sample]  # 返回列表，但只有一个样本
    
    def export_session(self, session_id: str, output_path: str, format: str = "json"):
        """
        导出单个session到文件
        
        Args:
            session_id: 会话ID
            output_path: 输出文件路径
            format: 输出格式 ("json" 或 "jsonl")
        """
        print(f"📖 导出 session: {session_id}")
        
        samples = self.export_session_to_chat_format(session_id)
        
        if not samples:
            print(f"❌ Session {session_id} 没有生成任何样本")
            return
        
        # 🔥 修正文件扩展名
        output_file = Path(output_path)
        correct_suffix = ".jsonl" if format == "jsonl" else ".json"
        if output_file.suffix not in [".json", ".jsonl"]:
            # 如果扩展名不正确，强制使用正确的扩展名
            output_file = output_file.with_suffix(correct_suffix)
            print(f"⚠️  文件扩展名已自动修正为: {output_file}")
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据格式写入文件
        if format == "jsonl":
            # JSONL格式：每行一个JSON对象
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        else:
            # JSON格式：格式化的JSON数组
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 成功导出 {len(samples)} 个样本到 {output_file} (格式: {format})")
    
    def export_all_sessions(self, output_path: str, format: str = "json"):
        """
        导出所有sessions到单个文件
        
        Args:
            output_path: 输出文件路径
            format: 输出格式 ("json" 或 "jsonl")
        """
        print("📖 导出所有sessions...")
        
        # 获取所有唯一的session_id
        session_ids = self.db.lca_actions.distinct("session_id")
        
        print(f"找到 {len(session_ids)} 个sessions")
        
        all_samples = []
        
        for session_id in session_ids:
            print(f"  处理 session: {session_id}")
            samples = self.export_session_to_chat_format(session_id)
            all_samples.extend(samples)
            print(f"    生成 {len(samples)} 个样本")
        
        if not all_samples:
            print("❌ 没有生成任何样本")
            return
        
        # 🔥 修正文件扩展名
        output_file = Path(output_path)
        correct_suffix = ".jsonl" if format == "jsonl" else ".json"
        if output_file.suffix not in [".json", ".jsonl"]:
            # 如果扩展名不正确，强制使用正确的扩展名
            output_file = output_file.with_suffix(correct_suffix)
            print(f"⚠️  文件扩展名已自动修正为: {output_file}")
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据格式写入文件
        if format == "jsonl":
            # JSONL格式：每行一个JSON对象
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in all_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        else:
            # JSON格式：格式化的JSON数组
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 成功导出总计 {len(all_samples)} 个样本到 {output_file} (格式: {format})")
        print(f"   来自 {len(session_ids)} 个sessions")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="导出MongoDB中的工作台数据为Chat Template格式"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        help="导出指定的session ID"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="导出所有sessions"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出JSONL文件路径"
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default="mongodb://localhost:27017/",
        help="MongoDB连接URI"
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="lci_database",
        help="数据库名称"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="输出格式 (json: 格式化JSON数组, jsonl: 每行一个JSON对象)"
    )
    
    args = parser.parse_args()
    
    # 初始化导出器
    exporter = TrainingDataExporter(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name
    )
    
    # 执行导出
    if args.all:
        exporter.export_all_sessions(args.output, format=args.format)
    elif args.session_id:
        exporter.export_session(args.session_id, args.output, format=args.format)
    else:
        print("❌ 错误: 请指定 --session-id 或 --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
