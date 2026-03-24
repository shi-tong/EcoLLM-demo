"""
Qwen-Agent 工具定义
将我们的 LCA 工具注册为 Qwen-Agent 的 BaseTool

这些工具会被 Qwen-Agent 的 Assistant 自动调用和管理
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Union

# 解决嵌套事件循环问题
try:
    import nest_asyncio
    nest_asyncio.apply()
    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False

logger = logging.getLogger(__name__)


def run_async(coro):
    """安全地运行异步函数"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环已在运行，创建新任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"run_async 失败: {e}")
        # 尝试直接运行
        return asyncio.run(coro)

# 尝试导入 qwen-agent
try:
    from qwen_agent.tools.base import BaseTool, register_tool
    QWEN_AGENT_AVAILABLE = True
except ImportError:
    QWEN_AGENT_AVAILABLE = False
    logger.warning("qwen-agent 未安装")
    # 创建占位符
    class BaseTool:
        pass
    def register_tool(name):
        def decorator(cls):
            return cls
        return decorator


# 全局 tool_service 引用，由 QwenAgentServiceV2 设置
_tool_service = None
_current_session_id = None


def set_tool_service(tool_service, session_id: str = None):
    """设置工具服务引用"""
    global _tool_service, _current_session_id
    _tool_service = tool_service
    _current_session_id = session_id
    logger.info(f"✅ Qwen-Agent 工具服务已设置，session_id: {session_id}")


def get_tool_service():
    """获取工具服务"""
    return _tool_service


def get_session_id():
    """获取当前会话 ID"""
    return _current_session_id


@register_tool('search_document')
class SearchDocumentTool(BaseTool):
    """搜索文档工具"""
    
    description = "Search uploaded PDF document content. Use this to find specific information in the document."
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "Search query, e.g., 'energy consumption', 'material input', 'CO2 emission'",
            "required": True
        },
        {
            "name": "max_results",
            "type": "integer",
            "description": "Maximum number of results to return (default: 5)",
            "required": False
        }
    ]
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """执行文档搜索"""
        try:
            if isinstance(params, str):
                params = json.loads(params)
                
            query = params.get("query", "")
            max_results = params.get("max_results", 5)
            session_id = get_session_id()
            
            if not session_id:
                return json.dumps({"error": "No document session available"})
                
            tool_service = get_tool_service()
            if not tool_service:
                return json.dumps({"error": "Tool service not initialized"})
            
            # 调用实际的搜索方法
            result = run_async(
                tool_service.search_document(
                    session_id=session_id,
                    query=query,
                    max_results=max_results
                )
            )
            
            logger.info(f"🔍 search_document 执行完成，找到 {len(result.get('results', []))} 个结果")
            return json.dumps(result, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"search_document 执行失败: {e}")
            return json.dumps({"error": str(e)})


@register_tool('record_process_flow')
class RecordProcessFlowTool(BaseTool):
    """记录 LCI 流程数据工具"""
    
    description = "Record LCI (Life Cycle Inventory) flow data including inputs and outputs with quantities and units."
    parameters = [
        {
            "name": "flow_type",
            "type": "string",
            "description": "Flow type: 'Input' or 'Output'",
            "required": True
        },
        {
            "name": "category",
            "type": "string",
            "description": "Category: Raw Material, Process Energy, Post-processing Energy, Feedstock Energy, Gas, Cooling Media, Product, Recovered Material, Waste, Emission",
            "required": True
        },
        {
            "name": "name",
            "type": "string",
            "description": "Name of the flow, e.g., 'Electricity', 'Steel', 'CO2'",
            "required": True
        },
        {
            "name": "value",
            "type": "number",
            "description": "Quantity value",
            "required": True
        },
        {
            "name": "unit",
            "type": "string",
            "description": "Unit of measurement, e.g., 'kWh', 'kg', 'MJ'",
            "required": True
        },
        {
            "name": "selected_chunk",
            "type": "string",
            "description": "Source text from document for traceability",
            "required": False
        },
        {
            "name": "note",
            "type": "string",
            "description": "Additional notes, e.g., 'SLM machine', 'Atomization'",
            "required": False
        }
    ]
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """执行记录"""
        try:
            if isinstance(params, str):
                params = json.loads(params)
            
            session_id = get_session_id()
            if not session_id:
                return json.dumps({"error": "No document session available"})
            
            tool_service = get_tool_service()
            if not tool_service:
                return json.dumps({"error": "Tool service not initialized"})
            
            # 🔥 完整的参数名映射（兼容 LLM 可能使用的各种参数名）
            mapped_params = {
                "session_id": session_id,
                "flow_type": params.get("flow_type") or params.get("type"),
                "category": params.get("category") or params.get("cat"),
                "name": params.get("name") or params.get("flow_name") or params.get("flow"),
                "value": params.get("value") or params.get("quantity") or params.get("amount") or params.get("val"),
                "unit": params.get("unit"),
                "note": params.get("note") or params.get("notes"),
                "selected_chunk": params.get("selected_chunk") or params.get("source") or params.get("chunk"),
            }
            
            # 移除 None 值，让方法使用默认值
            mapped_params = {k: v for k, v in mapped_params.items() if v is not None}
            
            result = run_async(
                tool_service.record_process_flow(**mapped_params)
            )
            
            logger.info(f"📝 record_process_flow 执行完成: {mapped_params.get('name')}")
            return json.dumps(result, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"record_process_flow 执行失败: {e}")
            return json.dumps({"error": str(e)})


@register_tool('record_parameter')
class RecordParameterTool(BaseTool):
    """记录中间参数工具"""
    
    description = "Record intermediate parameters for calculations, such as power, time, efficiency values."
    parameters = [
        {
            "name": "parameter_name",
            "type": "string",
            "description": "Name of the parameter, e.g., 'power', 'printing_time', 'efficiency'",
            "required": True
        },
        {
            "name": "value",
            "type": "number",
            "description": "Parameter value",
            "required": True
        },
        {
            "name": "unit",
            "type": "string",
            "description": "Unit of measurement, e.g., 'W', 'h', '%'",
            "required": True
        },
        {
            "name": "selected_chunk",
            "type": "string",
            "description": "Source text from document",
            "required": False
        },
        {
            "name": "note",
            "type": "string",
            "description": "Additional notes",
            "required": False
        }
    ]
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """执行记录"""
        try:
            if isinstance(params, str):
                params = json.loads(params)
            
            session_id = get_session_id()
            if not session_id:
                return json.dumps({"error": "No document session available"})
            
            tool_service = get_tool_service()
            if not tool_service:
                return json.dumps({"error": "Tool service not initialized"})
            
            # 🔥 完整的参数名映射（兼容 LLM 可能使用的各种参数名）
            # tool_service.record_parameter 期望: parameter_name, parameter_value, parameter_unit, selected_chunk
            
            # 处理 selected_chunk：可能是字符串或字典，且是必需参数
            chunk_raw = params.get("selected_chunk") or params.get("source") or params.get("chunk") or ""
            if isinstance(chunk_raw, str):
                selected_chunk = {"content": chunk_raw}
            elif isinstance(chunk_raw, dict):
                selected_chunk = chunk_raw
            else:
                selected_chunk = {"content": ""}
            
            mapped_params = {
                "session_id": session_id,
                "parameter_name": params.get("parameter_name") or params.get("name"),
                "parameter_value": params.get("value") or params.get("val") or params.get("parameter_value"),
                "parameter_unit": params.get("unit") or params.get("parameter_unit"),
                "note": params.get("note") or params.get("notes"),
                "selected_chunk": selected_chunk,
            }
            
            # 移除 None 值（但保留 selected_chunk）
            mapped_params = {k: v for k, v in mapped_params.items() if v is not None or k == "selected_chunk"}
            
            result = run_async(
                tool_service.record_parameter(**mapped_params)
            )
            
            logger.info(f"📝 record_parameter 执行完成: {mapped_params.get('parameter_name')}")
            return json.dumps(result, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"record_parameter 执行失败: {e}")
            return json.dumps({"error": str(e)})


@register_tool('define_lca_scope')
class DefineLCAScopeTool(BaseTool):
    """定义 LCA 范围工具"""
    
    description = "Define LCA scope parameters: Functional Unit, System Boundary, or Geographical Scope"
    parameters = [
        {
            "name": "parameter_name",
            "type": "string",
            "description": "Parameter type: 'Functional Unit', 'System Boundary', or 'Geographical Scope'",
            "required": True
        },
        {
            "name": "description",
            "type": "string",
            "description": "Full description of the parameter, e.g., '1 kg of 316L impeller manufactured by SLM'",
            "required": True
        },
        {
            "name": "value",
            "type": "number",
            "description": "Numeric value (optional, mainly for Functional Unit)",
            "required": False
        },
        {
            "name": "unit",
            "type": "string",
            "description": "Unit of the value (optional)",
            "required": False
        },
        {
            "name": "selected_chunk",
            "type": "string",
            "description": "Source text from document",
            "required": False
        },
        {
            "name": "note",
            "type": "string",
            "description": "Additional notes",
            "required": False
        }
    ]
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """执行定义"""
        try:
            if isinstance(params, str):
                params = json.loads(params)
            
            session_id = get_session_id()
            if not session_id:
                return json.dumps({"error": "No document session available"})
            
            tool_service = get_tool_service()
            if not tool_service:
                return json.dumps({"error": "Tool service not initialized"})
            
            # 🔥 完整的参数名映射（兼容 LLM 可能使用的各种参数名）
            mapped_params = {
                "session_id": session_id,
                "parameter_name": params.get("parameter_name") or params.get("name") or params.get("type"),
                "description": params.get("description") or params.get("desc") or params.get("content"),
                "value": params.get("value") or params.get("val"),
                "unit": params.get("unit"),
                "note": params.get("note") or params.get("notes"),
                "selected_chunk": params.get("selected_chunk") or params.get("source") or params.get("chunk"),
            }
            
            # 移除 None 值，让方法使用默认值
            mapped_params = {k: v for k, v in mapped_params.items() if v is not None}
            
            result = run_async(
                tool_service.define_lca_scope(**mapped_params)
            )
            
            logger.info(f"📝 define_lca_scope 执行完成: {mapped_params.get('parameter_name')}")
            return json.dumps(result, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"define_lca_scope 执行失败: {e}")
            return json.dumps({"error": str(e)})


@register_tool('get_session_summary')
class GetSessionSummaryTool(BaseTool):
    """获取会话摘要工具"""
    
    description = "Get a summary of all recorded data in the current session, including LCI flows and parameters."
    parameters = []
    
    def call(self, params: Union[str, dict] = None, **kwargs) -> str:
        """获取摘要"""
        try:
            session_id = get_session_id()
            if not session_id:
                return json.dumps({"error": "No document session available"})
                
            tool_service = get_tool_service()
            if not tool_service:
                return json.dumps({"error": "Tool service not initialized"})
            
            result = run_async(
                tool_service.get_session_summary(session_id=session_id)
            )
            
            logger.info(f"📊 get_session_summary 执行完成")
            return json.dumps(result, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"get_session_summary 执行失败: {e}")
            return json.dumps({"error": str(e)})


@register_tool('execute_calculation')
class ExecuteCalculationTool(BaseTool):
    """执行计算工具"""
    
    description = "Execute calculations on recorded parameters using formulas."
    parameters = [
        {
            "name": "formula",
            "type": "string",
            "description": "Calculation formula, e.g., 'energy_per_part = total_energy / num_parts'",
            "required": True
        },
        {
            "name": "result_name",
            "type": "string",
            "description": "Name for the calculation result",
            "required": True
        },
        {
            "name": "result_unit",
            "type": "string",
            "description": "Unit for the result",
            "required": True
        }
    ]
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """执行计算"""
        try:
            if isinstance(params, str):
                params = json.loads(params)
            
            session_id = get_session_id()
            if not session_id:
                return json.dumps({"error": "No document session available"})
            
            tool_service = get_tool_service()
            if not tool_service:
                return json.dumps({"error": "Tool service not initialized"})
            
            # 🔥 完整的参数名映射
            mapped_params = {
                "session_id": session_id,
                "formula": params.get("formula") or params.get("expression") or params.get("calc"),
                "result_name": params.get("result_name") or params.get("name") or params.get("output_name"),
                "result_unit": params.get("result_unit") or params.get("unit") or params.get("output_unit"),
            }
            
            # 移除 None 值
            mapped_params = {k: v for k, v in mapped_params.items() if v is not None}
            
            result = run_async(
                tool_service.execute_calculation(**mapped_params)
            )
            
            logger.info(f"🔢 execute_calculation 执行完成: {mapped_params.get('result_name')}")
            return json.dumps(result, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"execute_calculation 执行失败: {e}")
            return json.dumps({"error": str(e)})


# 获取所有工具类列表
def get_lca_tools() -> List[type]:
    """获取所有 LCA 工具类"""
    if not QWEN_AGENT_AVAILABLE:
        return []
    return [
        SearchDocumentTool,
        RecordProcessFlowTool,
        RecordParameterTool,
        DefineLCAScopeTool,
        GetSessionSummaryTool,
        ExecuteCalculationTool,
    ]


def get_lca_tool_names() -> List[str]:
    """获取所有 LCA 工具名称"""
    return [
        'search_document',
        'record_process_flow',
        'record_parameter',
        'define_lca_scope',
        'get_session_summary',
        'execute_calculation',
    ]
