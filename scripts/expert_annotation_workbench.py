#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LCA-LLM 专家标注工作台
三栏式专业界面，优化专家标注工作流程

Author: LCA-LLM Team
Version: 1.2 (Intent & Link_to 修复 - 2025-11-04)
Last Updated: 2025-11-04 13:00 UTC
"""

import streamlit as st
import requests
import base64
import tempfile
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加前端和后端组件路径
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.append(frontend_path)
sys.path.append(backend_path)

# 单位选择器组件不可用，直接设置为False
UNIT_SELECTOR_AVAILABLE = False

def get_unit_dropdown_options():
    """获取单位下拉选项"""
    import yaml
    
    try:
        # 读取标准单位库
        units_file = os.path.join(os.path.dirname(__file__), '..', 'resources', 'standard_units.yml')
        with open(units_file, 'r', encoding='utf-8') as f:
            units_data = yaml.safe_load(f)
        
        # 提取所有单位
        options = []
        for category_name, units_list in units_data.items():
            if isinstance(units_list, list):
                for unit_info in units_list:
                    if isinstance(unit_info, dict) and 'unit' in unit_info:
                        options.append({
                            'value': unit_info['unit'],
                            'label': unit_info.get('display_name', unit_info['unit']),
                            'category': unit_info.get('category', category_name)
                        })
        
        # 添加自定义选项
        options.append({'value': '__custom__', 'label': 'Custom'})
        
        return options
        
    except Exception as e:
        st.warning(f"无法加载单位库，使用备用单位: {e}")
        # 备用单位列表
        return [
            {'value': 'kg', 'label': 'kg'},
            {'value': 'L', 'label': 'L'},
            {'value': 'kWh', 'label': 'kWh'},
            {'value': 'm³', 'label': 'm³'},
            {'value': 'MJ', 'label': 'MJ'},
            {'value': 's', 'label': 's (second)'},
            {'value': 'min', 'label': 'min (minute)'},
            {'value': 'h', 'label': 'h (hour)'},
            {'value': 'pcs', 'label': 'pcs'},
            {'value': 't', 'label': 't'},
            {'value': 'GJ', 'label': 'GJ'},
            {'value': 'm²', 'label': 'm²'},
            {'value': '__custom__', 'label': 'Custom'}
        ]

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="EcoLLM Annotation Workbench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 全局样式 */
    .main > div {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* 栏目标题样式 */
    .column-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    
    /* Streamlit default slider styling - clean and modern */
    div[data-baseweb="slider"] > div > div {
        background-color: #f0f2f6 !important;
    }
    
    div[data-baseweb="slider"] > div > div > div {
        background-color: #ff4b4b !important;
    }
    
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    
    /* Table-like content formatting */
    .table-content {
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        overflow-x: auto;
        white-space: pre;
        line-height: 1.6;
    }
    
    .table-content table {
        border-collapse: collapse;
        width: 100%;
    }
    
    .table-content th, .table-content td {
        border: 1px solid #dee2e6;
        padding: 0.5rem;
        text-align: left;
    }
    
    .table-content th {
        background-color: #e9ecef;
        font-weight: bold;
    }
    
    /* 状态指示器 */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-active { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    
    /* 紧凑型度量卡片 */
    .metric-card {
        background: white;
        border-radius: 0.5rem;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# API配置
API_BASE_URL = "http://localhost:8000"

# ============================================================================
# 状态管理
# ============================================================================

def init_workbench_state():
    """初始化工作台状态"""
    if "workbench_session_id" not in st.session_state:
        st.session_state.workbench_session_id = None
    if "current_document" not in st.session_state:
        st.session_state.current_document = None
    if "current_context" not in st.session_state:
        st.session_state.current_context = None
    if "context_history" not in st.session_state:
        st.session_state.context_history = []
    if "extraction_log" not in st.session_state:
        st.session_state.extraction_log = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "current_search_context" not in st.session_state:
        st.session_state.current_search_context = []
    if "session_summary" not in st.session_state:
        st.session_state.session_summary = None
    if "last_action_id" not in st.session_state:
        st.session_state.last_action_id = None

def get_status_indicator(condition: bool) -> str:
    """获取状态指示器HTML"""
    status_class = "status-active" if condition else "status-error"
    return f'<span class="status-indicator {status_class}"></span>'

def get_last_action(session_id: str) -> dict:
    """获取session的最后一个action"""
    try:
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')
        db = client['lci_database']
        
        last_action = db.lca_actions.find_one(
            {'session_id': session_id},
            sort=[('created_at', -1)]
        )
        return last_action
    except Exception as e:
        print(f"Error getting last action: {e}")
        return None

def is_table_content(content: str) -> bool:
    """检测内容是否为表格格式"""
    lines = content.strip().split('\n')
    if len(lines) < 2:
        return False
    
    # 检查是否包含"## Table"标题
    if any('## Table' in line or '# Table' in line for line in lines[:3]):
        return True
    
    # 检查是否包含管道符分隔的表格
    pipe_lines = [line for line in lines if '|' in line and line.count('|') >= 3]
    if len(pipe_lines) >= 2:
        return True
    
    return False

def format_table_content(content: str) -> str:
    """将管道符分隔的表格内容格式化为HTML表格"""
    lines = content.strip().split('\n')
    
    # 只提取包含管道符的完整行，忽略被分割的内容
    table_lines = []
    
    for line in lines:
        # 只处理包含足够管道符的行（至少3个管道符，即4列）
        if '|' in line and line.count('|') >= 3:
            cleaned = line.strip()
            if cleaned.startswith('|'):
                cleaned = cleaned[1:]
            if cleaned.endswith('|'):
                cleaned = cleaned[:-1]
            table_lines.append(cleaned)
    
    if len(table_lines) < 2:
        return content
    
    # 构建HTML表格
    html_parts = ['<div style="overflow-x: auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.9em; margin: 1rem 0;">']
    
    is_header = True
    tbody_opened = False
    
    for line in table_lines:
        cells = [cell.strip() for cell in line.split('|')]
        
        # 跳过分隔线
        if all(set(cell.strip()) <= {'-', ' ', ':'} for cell in cells if cell.strip()):
            if is_header and not tbody_opened:
                html_parts.append('</tr></thead><tbody>')
                tbody_opened = True
                is_header = False
            continue
        
        if is_header:
            html_parts.append('<thead><tr>')
            for cell in cells:
                html_parts.append(f'<th style="border: 1px solid #dee2e6; padding: 0.75rem; background-color: #e9ecef; font-weight: bold; text-align: left;">{cell}</th>')
        else:
            if not tbody_opened:
                html_parts.append('</tr></thead><tbody>')
                tbody_opened = True
            html_parts.append('<tr>')
            for cell in cells:
                html_parts.append(f'<td style="border: 1px solid #dee2e6; padding: 0.75rem; vertical-align: top;">{cell}</td>')
            html_parts.append('</tr>')
    
    if not tbody_opened:
        html_parts.append('</tr></thead>')
    else:
        html_parts.append('</tbody>')
    
    html_parts.append('</table></div>')
    
    return ''.join(html_parts)

# ============================================================================
# API通信
# ============================================================================

def call_api(endpoint: str, method: str = "POST", data: dict = None, files: dict = None):
    """统一API调用函数"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        else:
            response = requests.get(url, params=data)
        
        response.raise_for_status()
        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json(),
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "data": None,
            "error": str(e)
        }

def log_extraction_step(tool: str, result: dict, details: str = ""):
    """记录提取步骤"""
    log_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "tool": tool,
        "success": result.get("success", False),
        "details": details,
        "context_id": st.session_state.current_context.get("chunk_id") if st.session_state.current_context else None
    }
    st.session_state.extraction_log.append(log_entry)

# ============================================================================
# 栏目A: 会话与实时摘要
# ============================================================================

def column_a_session_management():
    """栏目A: 会话管理和实时摘要"""
    
    # 栏目标题
    st.markdown('<div class="column-header">Session & Summary</div>', unsafe_allow_html=True)
    
    # 会话状态
    session_active = st.session_state.workbench_session_id is not None
    doc_loaded = st.session_state.current_document is not None
    context_selected = st.session_state.current_context is not None
    
    st.markdown("### Session Status")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"""
        {get_status_indicator(session_active)} **Session**  
        {get_status_indicator(doc_loaded)} **Document**  
        {get_status_indicator(context_selected)} **Context**
        """, unsafe_allow_html=True)
    
    with col2:
        if session_active:
            st.code(f"ID: {st.session_state.workbench_session_id[:8]}...")
        if doc_loaded:
            st.caption(f"Document: {st.session_state.current_document.get('filename', 'Unknown')}")
        if context_selected:
            st.caption(f"Context: {len(st.session_state.current_context.get('content', ''))} chars")
    
    # 会话控制
    st.markdown("### Session Control")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(" Record Preview", use_container_width=True, disabled=not session_active, 
                    help="Record chunk 0 & 1 for Context-Aware Retrieval training"):
            if session_active:
                with st.spinner("Recording document preview..."):
                    result = call_api(
                        "/tools/record-document-preview",
                        method="POST",
                        data={"session_id": st.session_state.workbench_session_id}
                    )
                    if result["success"]:
                        response_data = result.get("data", {})
                        preview_data = response_data.get("data", {})
                        st.success(f" Preview recorded! Chunk 0: {preview_data.get('chunk_0_length', 0)} chars, Chunk 1: {preview_data.get('chunk_1_length', 0)} chars")
                    else:
                        st.error(f"Failed: {result.get('error')}")
    
    with col2:
        if st.button("Reset Session", use_container_width=True):
            # 重置所有状态
            for key in ["workbench_session_id", "current_document", "current_context", 
                       "context_history", "extraction_log", "search_results", "current_search_context", "session_summary"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col3:
        if st.button("Get Summary", use_container_width=True, disabled=not session_active, key="get_summary_btn"):
            if session_active:
                with st.spinner("Getting summary..."):
                    # 使用GET方法调用session-summary端点（专家工作台使用 json 格式 + workbench 视图）
                    result = call_api(f"/tools/session-summary/{st.session_state.workbench_session_id}?format=json&view=workbench", method="GET")
                    if result["success"]:
                        st.session_state.session_summary = result["data"]
                        st.success("Summary updated")
                    else:
                        st.error(f"Failed: {result.get('error')}")
    
    with col4:
        if st.button("Record Check", use_container_width=True, disabled=not session_active, key="record_summary_check_btn", 
                    help="Record this summary check for SFT training data"):
            st.session_state.show_summary_check_dialog = True
    
    # 记录对话框（在按钮外部）
    if st.session_state.get("show_summary_check_dialog", False) and session_active:
        with st.form("summary_check_form", clear_on_submit=True):
            st.markdown("### Check Session Summary")
            st.info("Recording this action for training data")
            
            col_submit, col_cancel = st.columns(2)
            with col_submit:
                submitted = st.form_submit_button("Check Summary", use_container_width=True)
            with col_cancel:
                cancelled = st.form_submit_button("Cancel", use_container_width=True)
            
            if submitted:
                with st.spinner("Recording summary check..."):
                    record_payload = {
                        "session_id": st.session_state.workbench_session_id,
                        "rationale": "",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    record_result = call_api("/actions/record-summary-check", method="POST", data=record_payload)
                    
                    if record_result and record_result.get("success"):
                        st.success(f"Summary check recorded at {record_result.get('data', {}).get('timestamp', 'N/A')}")
                        
                        # 自动刷新 session_summary
                        summary_result = call_api(f"/tools/session-summary/{st.session_state.workbench_session_id}?format=json&view=workbench", method="GET")
                        if summary_result and summary_result.get("success"):
                            st.session_state.session_summary = summary_result.get("data")
                        
                        st.session_state.show_summary_check_dialog = False
                        st.rerun()
                    else:
                        st.error(f"Failed to record: {record_result.get('error') if record_result else 'Unknown error'}")
            
            if cancelled:
                st.session_state.show_summary_check_dialog = False
                st.rerun()
    
    # 实时摘要显示
    if st.session_state.session_summary:
        st.markdown("### Real-time Summary")
        # 🔥 添加空值检查，防止 AttributeError
        summary_data = st.session_state.session_summary.get("data", {}) if st.session_state.session_summary else {}
        
        # 如果 summary_data 是 None，设置为空字典
        if summary_data is None:
            summary_data = {}
        
        # 从新的数据结构中提取统计信息
        statistics = summary_data.get("statistics", {}) if isinstance(summary_data, dict) else {}
        decision_chain = summary_data.get("decision_chain", {}) if isinstance(summary_data, dict) else {}
        calculation_analysis = summary_data.get("calculation_analysis", {}) if isinstance(summary_data, dict) else {}
        pivot_analysis = summary_data.get("pivot_analysis", {}) if isinstance(summary_data, dict) else {}
        smart_skip_analysis = summary_data.get("smart_skip_analysis", {}) if isinstance(summary_data, dict) else {}
        
        # 度量卡片 - 使用新的数据结构
        scope_count = statistics.get("total_scopes_defined", 0)
        flow_count = statistics.get("total_flows_recorded", 0)
        calc_count = calculation_analysis.get("total_calculations", 0)
        pivot_count = pivot_analysis.get("total_pivots", 0)
        skip_count = smart_skip_analysis.get("total_smart_skips", 0)
        
        if scope_count > 0 or flow_count > 0 or calc_count > 0 or pivot_count > 0 or skip_count > 0:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">{scope_count}</h3>
                    <p style="margin:0; font-size:0.8em;">Scope Items</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#ff7f0e;">{flow_count}</h3>
                    <p style="margin:0; font-size:0.8em;">Flow Records</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#9467bd;">{calc_count}</h3>
                    <p style="margin:0; font-size:0.8em;">Calculations</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#d62728;">{pivot_count}</h3>
                    <p style="margin:0; font-size:0.8em;">Pivot Queries</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#17becf;">{skip_count}</h3>
                    <p style="margin:0; font-size:0.8em;">Smart Skips</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 总记录数显示
            total_records = scope_count + flow_count + calc_count
            st.markdown(f"""
            <div style="text-align: center; margin-top: 10px;">
                <span style="font-size: 1.2em; font-weight: bold; color: #2ca02c;">
                    Total Actions: {total_records}
                </span>
                </div>
                """, unsafe_allow_html=True)
    
    # 提取日志
    st.markdown("### Extraction Log")
    if st.session_state.extraction_log:
        # 显示最近5条记录
        recent_logs = st.session_state.extraction_log[-5:]
        for log in reversed(recent_logs):
            status_icon = "✓" if log["success"] else "✗"
            st.caption(f"{log['timestamp']} {status_icon} {log['tool']} - {log['details']}")
    else:
        st.info("No extraction activities yet")

# ============================================================================
# 栏目B: 文档与上下文
# ============================================================================

def column_b_document_context():
    """栏目B: 文档浏览和上下文管理"""
    
    # 栏目标题
    st.markdown('<div class="column-header">Document & Context</div>', unsafe_allow_html=True)
    
    # 文档上传区域
    st.markdown("### Document Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload production process document for analysis",
        label_visibility="collapsed"
    )
    
    # 显示处理按钮的条件：有上传文件且（没有当前文档 或 当前文档数据为空）
    show_process_button = (uploaded_file and 
                          (not st.session_state.current_document or 
                           st.session_state.current_document.get("total_pages", 0) == 0))
    
    if show_process_button:
        button_text = "Re-process Document" if st.session_state.current_document else "Process Document"
        if st.button(button_text, use_container_width=True):
            with st.spinner("Processing document..."):
                # 编码文件
                file_content = base64.b64encode(uploaded_file.read()).decode()
                
                # 调用API
                result = call_api(
                    "/tools/process-document",
                    data={
                        "file_content": file_content,
                        "filename": uploaded_file.name
                    }
                )
                
                if result["success"]:
                    # call_api包装了响应，实际数据在result["data"]["data"]中
                    api_response = result.get("data", {})  # call_api的包装层
                    document_data = api_response.get("data", {})  # 真正的后端数据
                    
                    st.session_state.workbench_session_id = document_data.get("session_id")
                    st.session_state.current_document = {
                        "filename": uploaded_file.name,
                        "total_pages": document_data.get("total_pages", 0),
                        "total_chunks": document_data.get("total_chunks", 0),
                        "full_text": document_data.get("full_text", "")
                    }
                    
                    # 安全的session_id显示
                    if st.session_state.workbench_session_id:
                        session_display = st.session_state.workbench_session_id[:8]
                        st.success(f"Document processed! Session: {session_display}...")
                    else:
                        st.error("Session ID not found in response!")
                        st.json(result)  # 显示完整响应用于调试
                    
                    st.rerun()
                else:
                    st.error(f"Processing failed: {result.get('error')}")
    
    # 文档信息显示
    if st.session_state.current_document:
        doc = st.session_state.current_document
        st.markdown("### Document Info")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pages", doc.get("total_pages", 0))
        with col2:
            st.metric("Chunks", doc.get("total_chunks", 0))
        with col3:
            st.metric("Text Length", len(doc.get("full_text", "")))
    
    # 搜索界面
    st.markdown("### Search")
    
    if not st.session_state.workbench_session_id:
        st.info("Please upload and process a document first to enable search functionality.")
        return
        
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Search in document",
            value=st.session_state.get("current_search_query", ""),  # 🔥 修复: 显示当前搜索查询
            placeholder="Single: energy | Batch: electricity, energy, power",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("Search", use_container_width=True)
    
    # 搜索参数
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        max_results = st.select_slider("Max Results", [3, 5, 8, 10], value=5)
    with col2:
        min_similarity = st.select_slider("Min Similarity", [0.2, 0.3, 0.4, 0.5], value=0.3)
    with col3:
        batch_mode = st.checkbox("Batch", value=False, help="Enable batch search (separate queries with commas)")
    with col4:
        if st.button("💡", key="keyword_suggest_btn", help="Smart Keyword Suggestion (Core + Extended Sampling)", use_container_width=True):
            st.session_state["show_keyword_suggest_dialog"] = True
            st.rerun()
    
    # Keyword Suggestion Dialog
    if st.session_state.get("show_keyword_suggest_dialog", False):
        st.markdown("---")
        st.markdown("### Smart Keyword Suggestion")
        st.caption("Two-tier system: Core keywords (required) + Extended keywords (random sampling)")
        
        # LCI Category Selection
        categories = [
            "Function Unit",
            "Product",
            "Raw Material",
            "Process Energy",
            "Post-processing Energy",
            "Feedstock Energy",
            "Gas",
            "Cooling Media",
            "Recovered Material",
            "Waste",
            "Emission"
        ]
        
        col_category, col_action = st.columns([3, 1])
        with col_category:
            selected_category = st.selectbox(
                "Select LCI Category",
                categories,
                index=0,  # Default: Function Unit (first to annotate)
                key="kw_category_select"
            )
        with col_action:
            st.write("")  # Alignment
            st.write("")  # Alignment
            if st.button("Generate", key="generate_keywords", use_container_width=True):
                # Call API for keyword suggestions
                with st.spinner("Generating..."):
                    result = call_api("/keywords/suggest", data={
                        "category": selected_category,
                        "min_keywords": 5,
                        "max_keywords": 8,
                        "extended_count": None  # Random 1-2
                    })
                    
                    if result.get("success", False):
                        keywords_list = result.get("data", {}).get("keywords", [])
                        breakdown = result.get("data", {}).get("breakdown", {})
                        
                        # Convert keyword list to comma-separated string
                        keywords_str = ", ".join(keywords_list)
                        st.session_state["suggested_keywords"] = keywords_str
                        st.session_state["keyword_breakdown"] = breakdown
                        
                        st.success(f"Generated {len(keywords_list)} keywords")
                    else:
                        st.error(f"Generation failed: {result.get('error', 'Unknown error')}")
        
        # Display suggested keywords
        if "suggested_keywords" in st.session_state and st.session_state["suggested_keywords"]:
            st.markdown("#### Suggested Keywords")
            
            # Display breakdown
            if "keyword_breakdown" in st.session_state:
                breakdown = st.session_state["keyword_breakdown"]
                col_core, col_ext = st.columns(2)
                with col_core:
                    st.markdown("**Core (Required)**")
                    for kw in breakdown.get("core", []):
                        st.markdown(f"- `{kw}`")
                with col_ext:
                    st.markdown("**Extended (Sampled)**")
                    for kw in breakdown.get("extended", []):
                        st.markdown(f"- `{kw}`")
            
            # Display full keyword string (editable)
            st.text_area(
                "Full Keywords (Editable)",
                value=st.session_state["suggested_keywords"],
                height=100,
                key="editable_keywords"
            )
            
            # Action buttons
            col_apply, col_regenerate, col_cancel = st.columns([1, 1, 1])
            with col_apply:
                if st.button("Apply", key="apply_keywords", use_container_width=True):
                    # Apply edited keywords to search box
                    st.session_state["current_search_query"] = st.session_state["editable_keywords"]
                    st.session_state["show_keyword_suggest_dialog"] = False
                    # Clean up temp state
                    if "suggested_keywords" in st.session_state:
                        del st.session_state["suggested_keywords"]
                    if "keyword_breakdown" in st.session_state:
                        del st.session_state["keyword_breakdown"]
                    st.success("Applied! Keywords filled in search box")
                    st.rerun()
            with col_regenerate:
                if st.button("Regenerate", key="regenerate_keywords", use_container_width=True):
                    # Clear old suggestions and regenerate
                    if "suggested_keywords" in st.session_state:
                        del st.session_state["suggested_keywords"]
                    if "keyword_breakdown" in st.session_state:
                        del st.session_state["keyword_breakdown"]
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key="cancel_keywords", use_container_width=True):
                    st.session_state["show_keyword_suggest_dialog"] = False
                    # Clean up temp state
                    if "suggested_keywords" in st.session_state:
                        del st.session_state["suggested_keywords"]
                    if "keyword_breakdown" in st.session_state:
                        del st.session_state["keyword_breakdown"]
                    st.rerun()
        
        st.markdown("---")
    
    if search_query and search_button:
        with st.spinner("Searching..."):
            # 判断是否为批量模式
            if batch_mode:
                # 解析逗号分隔的查询
                queries = [q.strip() for q in search_query.split(",") if q.strip()]
                api_data = {
                    "session_id": st.session_state.workbench_session_id,
                    "queries": queries,  # 批量模式
                    "max_results_per_query": max(2, max_results // len(queries)) if queries else 3,
                    "max_total_results": max_results,  # 使用用户选择的值
                    "extract_mode": "comprehensive",
                    "min_similarity": min_similarity,
                    "deduplicate": True
                }
            else:
                # 单查询模式
                api_data = {
                    "session_id": st.session_state.workbench_session_id,
                    "query": search_query,  # 单查询模式
                    "max_results": max_results,
                    "extract_mode": "comprehensive",
                    "min_similarity": min_similarity
                }
            
            result = call_api("/tools/search-document", data=api_data)
            
            if result["success"]:
                # call_api包装了响应，实际数据在result["data"]["data"]中
                api_response = result.get("data", {})  # call_api的包装层
                
                # 检查api_response是否为None
                if api_response is None:
                    st.error("Search response data is empty")
                    st.session_state.search_results = []
                    return
                
                search_data = api_response.get("data", {})  # 真正的搜索结果数据
                
                # 检查search_data是否为None
                if search_data is None:
                    st.error("Search results are empty")
                    st.session_state.search_results = []
                    return
                
                st.session_state.search_results = search_data.get("results", [])
                
                # 🔥 CRITICAL: 根据模式保存正确的查询格式
                # 批量模式：存储为数组（与LLM调用格式一致）
                # 单查询模式：存储为字符串
                if batch_mode and "queries" in search_data:
                    st.session_state.current_search_query = search_data.get("queries", [])  # 存储为数组
                else:
                    st.session_state.current_search_query = search_query  # 存储为字符串
                
                # 显示成功消息（区分单查询和批量查询）
                if batch_mode and "queries" in search_data:
                    queries_list = search_data.get("queries", [])
                    max_possible = len(queries_list) * 3  # max_results_per_query = 3
                    actual_results = len(st.session_state.search_results)
                    deduped_count = max_possible - actual_results
                    
                    st.success(f"Batch search completed: {len(queries_list)} queries, {actual_results} results found")
                    st.caption(f"Queries: {', '.join(queries_list)}")
                    if deduped_count > 0:
                        st.info(f"Deduplication removed {deduped_count} duplicate chunks")
                else:
                    st.success(f"Found {len(st.session_state.search_results)} relevant results")
            else:
                st.error(f"Search failed: {result.get('error')}")
                st.session_state.search_results = []
    
    # 搜索结果显示
    if st.session_state.search_results:
        # 搜索结果标题和操作按钮
        col_title, col_skip, col_pivot = st.columns([2, 1, 1])
        with col_title:
            st.markdown("### Search Results")
        with col_skip:
            if st.button("🔄 Smart Skip", key="smart_skip_search", help="Data already recorded, skip this category"):
                st.session_state["show_smart_skip_dialog"] = True
                st.rerun()
        with col_pivot:
            if st.button("❌ Pivot Query", key="pivot_search", help="Search failed, change keywords"):
                st.session_state["show_pivot_dialog"] = True
                st.rerun()
        
        for i, result_item in enumerate(st.session_state.search_results):
            similarity = result_item.get('similarity_score', result_item.get('similarity', 0))
            chunk_id = result_item.get('chunk_id', 'NO_CHUNK_ID')
            
            # Less is More: 移除厚重容器，使用简洁分隔线
            if i > 0:
                st.divider()  # 细线分隔，替代圆角框
            
            # 标题行：结果编号 + Chunk ID
            st.markdown(f"**Result {i+1}** - Chunk ID: `{chunk_id}`")
            
            # 内容显示（完整内容）
            full_content = result_item.get('content', '')
            
            # 检测是否为表格内容
            if is_table_content(full_content):
                with st.expander(f"Table Content ({len(full_content)} chars)", expanded=True):
                    # 显示格式化的表格
                    table_html = format_table_content(full_content)
                    st.markdown(table_html, unsafe_allow_html=True)
                    st.text(full_content)
            else:
                # 普通文本内容
                with st.expander(f"Content ({len(full_content)} chars)", expanded=False):
                    st.text(full_content)
            
            # 操作按钮
            if st.button("Use This Chunk", key=f"use_chunk_{i}", use_container_width=True):
                new_context = {
                    "content": result_item.get('content', ''),
                    "chunk_id": result_item.get('chunk_id', f"fallback_chunk_{i}"),
                    "similarity": similarity,
                    "timestamp": datetime.now().isoformat()
                }
                
                st.session_state.current_context = new_context
                
                # 保存完整的搜索上下文
                st.session_state.current_search_context = [
                    {
                        "chunk_id": item.get('chunk_id', f"fallback_chunk_{idx}"),
                        "content": item.get('content', ''),
                        "score": item.get('similarity_score', item.get('similarity', 0))
                    }
                    for idx, item in enumerate(st.session_state.search_results)
                ]
                
                # 保存选中的chunk信息
                st.session_state.selected_chunk_info = {
                    "chunk_id": result_item.get('chunk_id', f"fallback_chunk_{i}"),
                    "content": result_item.get('content', '')
                }
                
                # 添加到历史记录
                st.session_state.context_history.append(new_context)
                if len(st.session_state.context_history) > 10:
                    st.session_state.context_history.pop(0)
                
                st.success(f"Context set to Chunk {chunk_id}")
                st.rerun()
            
                
        
        # Pivot Query - Record Failure and Continue
        if st.session_state.get("show_pivot_dialog", False):
            st.markdown("---")
            st.markdown("### Record Search Failure")
            st.info("Document that this search didn't yield useful results")
            
            # 🔥 Record Button
            col_record, col_cancel = st.columns([1, 1])
            with col_record:
                if st.button("Record Failure and Continue", key="record_failure"):
                    # 获取当前搜索相关的变量
                    current_results = st.session_state.search_results
                    current_query = st.session_state.current_search_query
                    last_action_id = st.session_state.get("last_action_id")
                    
                    # 直接使用原始格式（字符串或数组）
                    failed_query = current_query if current_query else ""
                    
                    # 转换search_results为API所需的格式
                    failed_context_for_api = []
                    if current_results:
                        for result in current_results:
                            failed_context_for_api.append({
                                "chunk_id": result.get("chunk_id", ""),
                                "content": result.get("content", "")
                                # 🔥 不包含 score 字段：让 LLM 自己学习评估 chunk 质量
                            })
                    
                    with st.spinner("📝 Recording failure action..."):
                        # 调用 record_failure API
                        result = call_api(
                            "/tools/record-failure",
                            data={
                                "session_id": st.session_state.workbench_session_id,
                                "link_to": last_action_id,
                                "failed_query": failed_query,
                                "failed_context": failed_context_for_api
                            }
                        )
                        
                        if result.get("success"):
                            # 🔥 FIX: 正确的API响应路径 call_api -> data -> data -> new_action_id
                            api_response = result.get("data", {})
                            data = api_response.get("data", {}) if api_response else {}
                            new_action_id = data.get("new_action_id") if data else None
                            st.success(f"✅ Failure recorded as action {new_action_id}")
                            
                            # 🔥 更新session状态
                            st.session_state["last_action_id"] = new_action_id
                            st.session_state["last_intent"] = "pivot_query"
                            
                            # 🔥 NOTE: 不需要设置 next_intent
                            # 下一次成功操作会自动检测 last_intent == "pivot_query"
                            # 并建立 link_to 到这个 pivot 记录，形成"失败-成功闭环"
                            
                            # 🔥 清空搜索结果，让专家手动开始新搜索
                            st.session_state.search_results = []
                            st.session_state.current_search_query = ""
                            st.session_state.current_search_context = []
                            
                            st.info("✅ Next step: please use the search box above to start a new search with different keywords.")
                            
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            st.error(f"❌ Failed to record failure: {error_msg}")
                    
                    # Close dialog
                    st.session_state["show_pivot_dialog"] = False
                    st.rerun()
            
            with col_cancel:
                if st.button("Cancel", key="cancel_pivot"):
                    st.session_state["show_pivot_dialog"] = False
                    st.rerun()
        
        # 🔥 NEW: Smart Skip - Record when data already extracted
        if st.session_state.get("show_smart_skip_dialog", False):
            st.markdown("---")
            st.markdown("### 🔄 Record Smart Skip")
            st.info("Skip a category: either data already recorded, or searched but not found in document.")
            
            # 🔥 修复：使用唯一的 form key 和 clear_on_submit
            # 使用时间戳确保每次打开 dialog 时 form key 都是唯一的
            if "smart_skip_form_counter" not in st.session_state:
                st.session_state.smart_skip_form_counter = 0
            
            form_key = f"smart_skip_form_{st.session_state.smart_skip_form_counter}"
            
            with st.form(key=form_key, clear_on_submit=True):
                # Category selection
                st.markdown("**Which category are you skipping?**")
                category = st.selectbox(
                    "Category:",
                    options=[
                        # 输入流（6个）
                        "Raw Material",
                        "Process Energy",
                        "Post-processing Energy",
                        "Feedstock Energy",
                        "Gas",
                        "Cooling Media",
                        # 输出流（4个）
                        "Product",
                        "Recovered Material",
                        "Waste",
                        "Emission"
                    ],
                    key=f"smart_skip_category_{st.session_state.smart_skip_form_counter}",
                    help="Select the category to skip"
                )
                
                # Skip reason selection
                skip_reason = st.selectbox(
                    "Skip reason:",
                    options=[
                        "already_recorded",
                        "not_found"
                    ],
                    key=f"smart_skip_reason_{st.session_state.smart_skip_form_counter}",
                    help="already_recorded: data extracted earlier | not_found: searched but not in document"
                )
                
                # Submit button
                col_submit, col_cancel = st.columns([1, 1])
                with col_submit:
                    submitted = st.form_submit_button("✅ Record Smart Skip", use_container_width=True)
                with col_cancel:
                    cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
            
            # 处理取消
            if cancel:
                st.session_state["show_smart_skip_dialog"] = False
                st.session_state.smart_skip_form_counter += 1  # 增加计数器，确保下次 form key 不同
                st.rerun()
            
            # 处理提交
            if submitted:
                    # 验证前置条件
                    session_id = st.session_state.get("workbench_session_id")
                    
                    if not session_id:
                        st.error("❌ No active session! Please upload a document first.")
                        st.stop()
                    
                    # 获取当前搜索相关的变量
                    current_results = st.session_state.search_results
                    last_action_id = st.session_state.get("last_action_id")
                    current_query = st.session_state.get("current_search_query", "")
                    
                    # 使用第一个搜索结果作为skipped_chunk（如果有的话）
                    skipped_chunk = None
                    if current_results and len(current_results) > 0:
                        first_result = current_results[0]
                        skipped_chunk = {
                            "chunk_id": first_result.get("chunk_id", ""),
                            "content": first_result.get("content", ""),
                            "score": first_result.get("similarity_score", first_result.get("similarity", 0))
                        }
                    
                    # 🔥 NEW: 构建完整的 search_context（所有搜索结果）
                    search_context = []
                    if current_results:
                        for result in current_results:
                            search_context.append({
                                "chunk_id": result.get("chunk_id", ""),
                                "content": result.get("content", ""),
                                "score": result.get("similarity_score", result.get("similarity", 0))
                            })
                    
                    with st.spinner("🔄 Recording smart skip..."):
                        # 调用smart skip API
                        result = call_api(
                            "/tools/record-smart-skip",
                            data={
                                "session_id": st.session_state.workbench_session_id,
                                "category": category,
                                "skip_reason": skip_reason,
                                "link_to": last_action_id,
                                "skipped_chunk": skipped_chunk,
                                "skip_rationale": None,
                                "search_context": search_context,
                                "search_query": current_query
                            }
                        )
                        
                        if result.get("success"):
                            api_response = result.get("data", {})
                            data = api_response.get("data", {}) if api_response else {}
                            new_action_id = data.get("new_action_id") if data else None
                            st.success(f"✅ Smart skip recorded as action {new_action_id}")
                            
                            # 更新session状态
                            st.session_state["last_action_id"] = new_action_id
                            st.session_state["last_intent"] = "smart_skip"
                            
                            # 清空搜索结果，继续下一个大类
                            st.session_state.search_results = []
                            st.session_state.current_search_query = ""
                            st.session_state.current_search_context = []
                            
                            st.info(f"✅ Category '{category}' skipped. Continue with the next category.")
                            
                            # 🔥 只在成功时才关闭 dialog 并 rerun
                            st.session_state["show_smart_skip_dialog"] = False
                            st.session_state.smart_skip_form_counter += 1
                            st.rerun()
                            
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            st.error(f"❌ Failed to record smart skip: {error_msg}")
                            # 失败时不关闭 dialog，让用户看到错误信息
    
    
    # 当前上下文显示
    if st.session_state.current_context:
        st.markdown("### Current Context")
        
        # Less is More: 移除圆角容器
        
        # 上下文信息和操作按钮
        ctx = st.session_state.current_context
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            st.caption(f"Chunk ID: {ctx.get('chunk_id', 'Unknown')}")
        with col2:
            st.caption(f"Content length: {len(ctx.get('content', ''))} chars")
        with col3:
            if st.button("Refine", key="refine_same", help="Extract more data from the same chunk"):
                # 设置refine_same模式
                st.session_state["refine_same_mode"] = True
                st.success("Refine mode activated! Extract another item from this chunk.")
                st.rerun()
        with col4:
            if st.button("Clear", key="clear_context"):
                st.session_state.current_context = None
                st.session_state["refine_same_mode"] = False
                # 清除备注输入框状态
                if "refine_note" in st.session_state:
                    del st.session_state["refine_note"]
                # 🔥 修复: 清除搜索相关状态，避免后续搜索时出现None错误
                if "search_results" in st.session_state:
                    st.session_state.search_results = []
                if "current_search_query" in st.session_state:
                    st.session_state.current_search_query = ""
                if "current_search_context" in st.session_state:
                    st.session_state.current_search_context = []
                st.rerun()
        
        # 上下文内容
        st.text_area(
            "Context Content",
            value=ctx.get('content', ''),
            height=150,
            disabled=True,
            label_visibility="collapsed"
        )
        
        # refine_same模式指示器
        if st.session_state.get("refine_same_mode", False):
            st.info("Refine Same mode: continuing extraction from current context")
            

# ============================================================================
# 栏目C: 统一提取工具
# ============================================================================

def column_c_extraction_tools():
    """栏目C: 统一数据提取工具"""
    
    # 栏目标题
    st.markdown('<div class="column-header">Extraction Tools</div>', unsafe_allow_html=True)
    
    # 检查前置条件 - 现在支持无上下文模式
    has_context = st.session_state.current_context is not None
    if not has_context:
        st.info("No Context Mode: some tools can work without context (e.g., Calculation, Flow recording for calculation results).")
    
    # 工具选择器 - Less is More: 移除不必要的容器
    st.markdown("### Tool Selection")
    
    extraction_tool = st.selectbox(
        "What type of information does this context contain?",
        ["Select Tool...", 
         "LCA Scope (Project Goals & Boundaries)", 
         "Process Flow (Materials & Emissions)",
         "Parameter Recorder (For Calculations)",
         "Calculation (Mathematical Operations)"],
        help="Select the appropriate tool based on the current context content."
    )
    
    # 根据选择显示相应工具
    if extraction_tool == "LCA Scope (Project Goals & Boundaries)":
        if not has_context:
            st.warning("LCA Scope typically requires context")
            st.info("LCA Scope extraction usually works best with document context. Consider selecting a context first, or use this tool only if you have specific scope data to record.")
        render_lca_scope_tool()
    elif extraction_tool == "Process Flow (Materials & Emissions)":
        render_process_flow_tool()
    elif extraction_tool == "Parameter Recorder (For Calculations)":
        if not has_context:
            st.error("Parameter Recorder requires context")
            st.info("Parameter Recorder is used to extract raw parameter values from documents. Please select a document chunk first.")
        else:
            render_parameter_tool()
    elif extraction_tool == "Calculation (Mathematical Operations)":
        render_calculation_tool()
    else:
        st.info("Please select an extraction tool to continue.")

def render_lca_scope_tool():
    """渲染LCA范围定义工具"""
    st.markdown("### LCA Scope Extraction")
    st.caption("Extract project goals and boundaries from the current context")
    
    # 三要素选择
    scope_element = st.selectbox(
        "Which scope element to define?",
        ["Select Element...", "Function Unit", "System Boundary", "Geographical Scope"]
    )
    
    # 专家决策变量初始化
    scope_rationale = None
    
    if scope_element != "Select Element...":
        with st.form(f"scope_form_{scope_element.lower().replace(' ', '_')}"):
            st.write(f"**{scope_element}**")
            
            # 根据不同的scope element显示不同的输入字段
            if scope_element == "Function Unit":
                # Function Unit只需要: Description
                description = st.text_area(
                    "Description", 
                    placeholder="Describe the function unit (e.g., '1 kg of product X for consumer use')",
                    height=100
                )
                # 设置默认值，不显示给用户
                value = 0.0
                unit = ""
                unit_valid = True
                        
            elif scope_element == "System Boundary":
                # System Boundary只需要: Description
                description = st.text_area(
                    "System Boundary Description", 
                    placeholder="Describe the system boundary (e.g., 'Cradle-to-gate: from raw material extraction to factory gate', 'Cradle-to-grave: full life cycle including use and disposal')",
                    height=120
                )
                # 设置默认值，不显示给用户
                value = 0.0
                unit = ""
                unit_valid = True
                
            elif scope_element == "Geographical Scope":
                # Geographical Scope只需要: Description  
                description = st.text_area(
                    "Geographical Scope Description", 
                    placeholder="Describe the geographical scope (e.g., 'China mainland', 'European Union', 'Global average', 'North America')",
                    height=120
                )
                # 设置默认值，不显示给用户
                value = 0.0
                unit = ""
                unit_valid = True
            
            
            submitted = st.form_submit_button(f"Define {scope_element}", use_container_width=True)
            
            if submitted:
                # 验证输入
                validation_passed = True
                if not description.strip():
                    st.error("Description is required")
                    validation_passed = False
                if UNIT_SELECTOR_AVAILABLE and unit and not unit_valid:
                    st.error("Please select a valid unit")
                    validation_passed = False
                
                if validation_passed:
                    with st.spinner("Defining scope..."):
                        # 🔥 NEW: 构建完整的决策逻辑数据（与record_process_flow保持一致）
                        
                        # 准备搜索上下文数据
                        search_context = []
                        selected_chunk = None
                        link_to = None
                        
                        # 准备搜索上下文数据
                        search_context = st.session_state.current_search_context if st.session_state.current_search_context else []
                        
                        # 构建选中的chunk
                        selected_chunk = None
                        if st.session_state.current_context:
                            selected_chunk = {
                                "chunk_id": st.session_state.current_context.get("chunk_id", ""),
                                "content": st.session_state.current_context.get("content", "")
                                # 🔥 不包含 score 字段：让 LLM 自己学习评估 chunk 质量
                            }
                        
                        # 🔥 确定意图类型
                        if st.session_state.get("refine_same_mode", False):
                            intent = "refine_same"
                        else:
                            intent = "select_best"
                        
                        # 🔥 NEW: 实现link_to逻辑链条 (按照新方案)
                        last_intent = st.session_state.get("last_intent")
                        last_action_id = st.session_state.get("last_action_id")
                        
                        link_to = None
                        
                        # 🔥 NEW: 正确的link_to逻辑 - 需要建立依赖的情况
                        if intent == "refine_same": 
                            # 情况1: refine_same链接到它正在精炼的原始"成功"动作
                            link_to = last_action_id
                        elif intent == "select_best" and last_intent == "pivot_query":
                            # 情况2: 失败-成功闭环 - 成功动作链接到它解决的失败动作
                            link_to = last_action_id
                        elif intent == "select_best" and not st.session_state.current_context and last_intent == "calculate":
                            # 情况3: 记录计算结果 (无上下文Scope) 链接到产生它的 Calculation 动作
                            link_to = last_action_id
                        
                        result = call_api(
                            "/tools/define-lca-scope",
                            data={
                                "session_id": st.session_state.workbench_session_id,
                                "parameter_name": scope_element,
                                "description": description,
                                "value": value if value > 0 else None,
                                "unit": unit if unit else None,
                                "source_content": st.session_state.current_context["content"] if st.session_state.current_context else None,
                                "note": None,  # LCA Scope暂不需要note
                                "search_query": st.session_state.get("current_search_query"),
                                "search_context": search_context,
                                "selected_chunk": selected_chunk,
                                "intent": intent,
                                "link_to": link_to
                            }
                        )
                        
                        log_extraction_step("define_lca_scope", result, f"{scope_element}: {description[:30]}...")
                        
                        if result["success"]:
                            st.success(f"{scope_element} defined successfully")
                            
                            # 🔥 NEW: 更新last_action_id和last_intent用于下次链接
                            api_response = result.get("data", {})
                            data = api_response.get("data", {}) if api_response else {}
                            action_id = data.get("action_id") if data else None
                            
                            if action_id:
                                st.info(f"Action ID: {action_id} (Intent: {intent})")
                                st.session_state["last_action_id"] = action_id
                                st.session_state["last_intent"] = intent
                            
                            # 重置refine_same模式（与Flow Tool保持一致）
                            if intent == "refine_same":
                                st.session_state["refine_same_mode"] = False
                                # 清除备注输入框状态
                                if "refine_note" in st.session_state:
                                    del st.session_state["refine_note"]
                            
                            # 清除会话摘要以强制刷新
                            if "session_summary" in st.session_state:
                                del st.session_state["session_summary"]
                        else:
                            st.error(f"Failed: {result.get('error')}")

def render_process_flow_tool():
    """渲染工艺流程记录工具"""
    st.markdown("### Process Flow Extraction")
    
    # 检查是否有上下文，显示不同的说明
    has_context = st.session_state.current_context is not None
    if has_context:
        st.caption("Extract materials, energy, and emissions from the current context")
    else:
        st.caption("Record calculation results or manual flow data (no context required)")
        st.info("No Context Mode: you can record calculation results or manually input flow data without selecting a context.")
    
    # 不使用form，改用普通布局以支持动态Custom Unit输入框
    col1, col2 = st.columns(2)
    
    with col1:
        # Note字段（可选）- 用于区分细节
        note = st.text_input(
            "Note (Optional)",
            placeholder="e.g., SLM machine, Atomization, Build chamber",
            help="Add a brief note to distinguish details (e.g., 'SLM machine' for electricity, 'Atomization' for energy)",
            key="pf_note"
        )
        
        name = st.text_input(
                "Material/Substance Name",
                placeholder="e.g., Ethylene",
                key="pf_name"
            )
        
        value = st.text_input("Quantity", placeholder="e.g., 18.99", key="pf_value")
        
        # 单位选择 - 使用完整的单位库
        selected_unit = None
        try:
            # 直接加载完整单位库
            options = get_unit_dropdown_options()
            
            # 提取单位值作为选项
            unit_options = []
            for opt in options:
                if opt['value'] == '__custom__':
                    unit_options.append("-- Other (Custom Unit) --")
                else:
                    unit_options.append(opt['value'])
            
            # 单位下拉选择
            selected_unit = st.selectbox(
                "Unit",
                options=unit_options,
                key="pf_unit_select"
            )
            
            # 先设置默认值
            unit = selected_unit
            unit_valid = True
                
        except Exception as e:
            # 备用方案：使用传统输入框
            st.warning(f"单位库加载失败，使用传统输入: {e}")
            unit = st.text_input("Unit", placeholder="e.g., kg, L, kWh", key="pf_unit_fallback")
            unit_valid = len(unit.strip()) > 0 if unit else False
        
    with col2:
        flow_type_display = st.selectbox("Flow Type", ["Input", "Output", "Emission"], key="pf_flow_type")
        flow_type_mapping = {"Input": "Input", "Output": "Output", "Emission": "Output"}
        flow_type = flow_type_mapping[flow_type_display]
        
        lci_category = st.selectbox("LCI Category", [
            # Phase 1: Input Flows
            "Raw Material",
            "Process Energy",         # 机器能耗（printing, laser, heater等）
            "Post-processing Energy", # 后处理能耗（heat treatment, machining等）
            "Feedstock Energy",       # 粉末制备能耗（atomization等）
            "Gas",                    # 气体（argon, nitrogen等）
            "Cooling Media",          # 冷却/加工液体（water, coolant, cutting fluid等）
            # Phase 2: Output Flows
            "Product",
            "Recovered Material",     # 回收材料（recovered powder等）
            "Waste",
            "Emission"                # 排放（VOC, particulate, wastewater等）
        ], key="pf_lci_category", help="For process parameters (time, temperature, etc.), use the Parameter Recorder tool instead")
        
        process_name = st.text_input("Process Name", placeholder="e.g., Polymerization", key="pf_process")
        cas_number = st.text_input("CAS Number", placeholder="Optional", key="pf_cas")
    
    # 🔥 REMOVED: expert_rationale 字段已废弃
    # 现在由 CAMEL AI 自动生成 reasoning，不需要手动填写 rationale
    
    # Custom Unit输入框 - 现在可以实时显示！
    if selected_unit and selected_unit == "-- Other (Custom Unit) --":
        st.markdown("#### Custom Unit")
        st.info("You selected a custom unit. Please enter it below:")
        custom_unit_input = st.text_input(
            "Enter your custom unit:",
            placeholder="e.g., kg CO₂-eq/MJ, L/min, pieces/hour",
            key="pf_custom_unit",
            help="Enter a custom unit that is not available in the dropdown list above"
        )
        # 更新unit变量
        if custom_unit_input.strip():
            unit = custom_unit_input.strip()
        else:
            unit = ""
    
    # 提交按钮
    st.markdown("---")
    submitted = st.button("Record Process Flow", use_container_width=True)

    if submitted:
        # 验证必填字段
        validation_passed = True
        if not name.strip():
            st.error("Material name is required")
            validation_passed = False
        
        # 安全转换value为float并验证
        try:
            value_float = float(value) if value and str(value).strip() else None
            if value_float is None or value_float <= 0:
                st.error("Quantity must be greater than 0")
                validation_passed = False
        except (ValueError, TypeError):
            st.error(f"Invalid quantity value: '{value}'. Please enter a valid number.")
            validation_passed = False
        
        # 单位验证 - 特殊处理Custom Unit
        if not unit or not unit.strip():
            st.error("Please enter or select a unit")
            validation_passed = False
        
        if validation_passed:
            with st.spinner("Recording flow..."):
                # 准备选择的chunk数据（不包含 score，让 LLM 自己评估）
                selected_chunk = None
                if st.session_state.current_context:
                    selected_chunk = {
                        "chunk_id": st.session_state.current_context.get("chunk_id", "unknown"),
                        "content": st.session_state.current_context.get("content", "")
                        # 🔥 不包含 score 字段：让 LLM 自己学习评估 chunk 质量
                    }
                
                # 准备搜索上下文数据
                search_context = st.session_state.current_search_context if st.session_state.current_search_context else None
                
                # 🔥 确定意图类型
                if st.session_state.get("refine_same_mode", False):
                    intent = "refine_same"
                else:
                    intent = "select_best"
                
                # 🔥 实现link_to逻辑链条
                last_intent = st.session_state.get("last_intent")
                last_action_id = st.session_state.get("last_action_id")
                
                link_to = None
                
                # 🔥 正确的link_to逻辑 - 需要建立依赖的情况
                if intent == "refine_same": 
                    # 情况1: refine_same链接到它正在精炼的原始"成功"动作
                    link_to = last_action_id
                elif intent == "select_best" and last_intent == "pivot_query":
                    # 情况2: 失败-成功闭环 - 成功动作链接到它解决的失败动作
                    link_to = last_action_id
                elif last_intent == "calculate":
                    # 情况3: 记录计算结果链接到产生它的 Calculation 动作
                    link_to = last_action_id
                
                # 🔥 NEW: 构建API请求数据
                api_data = {
                        "session_id": st.session_state.workbench_session_id,
                        "name": name,
                        "value": value_float,
                        "unit": unit,
                        "flow_type": flow_type,
                        "category": lci_category,
                        "cas_number": cas_number if cas_number else None,
                        "process_name": process_name if process_name else None,
                        "note": note if note and note.strip() else None,  # 🔥 NEW: 添加note字段
                    "search_query": st.session_state.get("current_search_query"),
                        "search_context": search_context,
                    "selected_chunk": selected_chunk,
                    "intent": intent,
                    "link_to": link_to
                }
                
                
                result = call_api("/tools/record-process-flow", data=api_data)
                
                log_extraction_step("record_process_flow", result, f"{name}: {value} {unit}")
                
                
                if result["success"]:
                    # 显示成功消息，区分不同意图
                    if intent == "refine_same":
                        st.success("Refined data recorded successfully from the same chunk!")
                    elif last_intent == "pivot_query":
                        # 🔥 FIX: 检测是否是 pivot 后的成功操作（通过 last_intent 判断）
                        st.success("🔄 Data recorded successfully after pivot!")
                    else:
                        st.success("Process flow recorded successfully")
                    
                    # call_api返回的结构: {"success": True, "data": response.json(), ...}
                    # response.json()的结构: {"success": true, "data": {...}, ...}
                    # 所以实际数据路径是: result["data"]["data"]
                    api_response = result.get("data", {})  # 这是API的完整响应
                    data = api_response.get("data", {}) if api_response else {}    # 安全检查
                    
                    # 调试信息
                    if not data:
                        st.error(f"Debug: API response structure: {result}")
                        st.error(f"Debug: api_response: {api_response}")
                    
                    flow_id = data.get("flow_id") if data else None
                    action_id = data.get("action_id") if data else None
                    
                    if flow_id:
                        st.info(f"Flow ID: {flow_id}")
                    if action_id:
                        st.info(f"Action ID: {action_id} (Intent: {intent})")
                        # 🔥 NEW: 更新last_action_id和last_intent用于下次链接
                        st.session_state["last_action_id"] = action_id
                        st.session_state["last_intent"] = intent
                    else:
                        st.error("No action_id found in API response!")
                        st.error(f"Debug: Available keys in data: {list(data.keys()) if data else 'data is empty'}")
                    
                    # 重置refine_same模式
                    if intent == "refine_same":
                        st.session_state["refine_same_mode"] = False
                        # 清除备注输入框状态
                        if "refine_note" in st.session_state:
                            del st.session_state["refine_note"]
                    
                    # 清除会话摘要以强制刷新
                    if "session_summary" in st.session_state:
                        del st.session_state["session_summary"]
                else:
                    st.error(f"Recording failed: {result.get('error')}")

def render_parameter_tool():
    """渲染参数记录工具 - 用于提取待用于计算的原始参数"""
    st.markdown("### Parameter Recorder")
    st.caption("Extract raw parameter values from documents for later calculations")
    
    st.info("📌 **Purpose**: Record original parameter values (like 'power = 10 kW' or 'time = 5 h') extracted from documents. These values will be used later in calculations.")
    
    # 获取当前上下文
    current_context = st.session_state.get("current_context")
    if not current_context:
        st.error("No context selected! Please select a document chunk first.")
        return
    
    # 参数信息输入
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Note字段（可选）- 用于区分细节
        note = st.text_input(
            "Note (Optional)",
            placeholder="e.g., SLM machine, Heat treatment",
            help="Add a brief note to distinguish details",
            key="param_note"
        )
        
        parameter_name = st.text_input(
            "Parameter Name *",
            help="Parameter name, e.g., 'power', 'printing_time', 'temperature'",
            placeholder="e.g., power",
            key="param_name"
        )
    
    with col2:
        parameter_value = st.text_input(
            "Parameter Value *",
            placeholder="e.g., 10.5",
            help="Numerical value extracted from the document"
        )
    
    # 单位选择 - 使用标准单位库（与Process Flow一致）
    selected_unit = None
    try:
        # 直接加载完整单位库
        options = get_unit_dropdown_options()
        
        # 提取单位值作为选项
        unit_options = []
        for opt in options:
            if opt['value'] == '__custom__':
                unit_options.append("-- Other (Custom Unit) --")
            else:
                unit_options.append(opt['value'])
        
        # 单位下拉选择
        selected_unit = st.selectbox(
            "Parameter Unit",
            options=unit_options,
            key="param_unit_select",
            help="Select from standard units or choose 'Other' for custom unit"
        )
        
        # 如果选择了Custom，显示输入框
        if selected_unit == "-- Other (Custom Unit) --":
            parameter_unit = st.text_input(
                "Custom Unit",
                placeholder="e.g., kW, °C, MPa",
                key="param_custom_unit"
            )
        else:
            parameter_unit = selected_unit
            
    except Exception as e:
        # 备用方案：使用传统输入框
        st.warning(f"单位库加载失败，使用传统输入: {e}")
        parameter_unit = st.text_input(
            "Parameter Unit",
            placeholder="e.g., kW, h, °C",
            key="param_unit_fallback"
        )
    
    # 自动检测intent（与Scope/Flow工具保持一致）
    st.markdown("---")
    if st.session_state.get("refine_same_mode", False):
        intent = "refine_same"
    else:
        intent = "select_best"
    
    # 显示当前intent状态
    intent_display = {
        "select_best": "Select Best - choosing from search results",
        "refine_same": "Refine Same - extracting more from the same context"
    }
    st.info(f"**Decision Type (Auto-detected)**: {intent_display.get(intent, intent)}")
    
    # 自动设置link_to（如果是refine_same）
    link_to = None
    if intent == "refine_same":
        last_action_id = st.session_state.get("last_action_id", "")
        if last_action_id:
            link_to = last_action_id
            st.info(f"**Auto-linking**: Will link to {last_action_id}")
    
    # Record按钮
    st.markdown("---")
    if st.button("Record Parameter", use_container_width=True):
        # 验证输入
        if not parameter_name or not parameter_name.strip():
            st.error("Parameter name is required!")
            return
        
        # 安全转换parameter_value为float
        try:
            value_float = float(parameter_value) if parameter_value and str(parameter_value).strip() else None
        except (ValueError, TypeError):
            st.error(f"Invalid parameter value: '{parameter_value}'. Please enter a valid number.")
            return
        
        # 准备数据（与Scope/Flow保持一致的结构）
        # expert_decision在顶层，不在selected_chunk中
        parameter_data = {
            "session_id": st.session_state.workbench_session_id,
            "parameter_name": parameter_name.strip(),
            "parameter_value": value_float,
            "parameter_unit": parameter_unit.strip() if parameter_unit else None,
            "selected_chunk": {
                "chunk_id": current_context.get("chunk_id"),
                "content": current_context.get("content")
                # 🔥 不包含 score 字段：让 LLM 自己学习评估 chunk 质量（后端会使用默认值 0.0）
            },
            "search_query": st.session_state.get("current_search_query"),
            "search_context": st.session_state.current_search_context if st.session_state.current_search_context else [],
            "intent": intent,
            "link_to": link_to if link_to else None,
            "note": note if note and note.strip() else None  # 🔥 NEW: 添加note字段
        }
        
        # 调用API
        with st.spinner("Recording parameter..."):
            result = call_api("/tools/record-parameter", method="POST", data=parameter_data)
            
            # 错误处理
            if not result:
                st.error("API call failed: response is empty")
                st.error("Please verify that the backend is running and the payload format is correct")
                st.stop()
            
            if result.get("success"):
                st.success("Parameter recorded successfully!")
                
                # 更新session state - 正确解析三层嵌套的API响应
                # API响应结构: {"data": {"data": {"new_action_id": "ACT_XXXX"}}}
                new_action_id = result.get("data", {}).get("data", {}).get("new_action_id")
                if new_action_id:
                    st.session_state["last_action_id"] = new_action_id
                    st.session_state["last_intent"] = intent
                    st.session_state["last_record_type"] = "parameter"
                
                # 重置refine_same模式（与Flow Tool保持一致）
                if intent == "refine_same":
                    st.session_state["refine_same_mode"] = False
                    # 清除备注输入框状态
                    if "refine_note" in st.session_state:
                        del st.session_state["refine_note"]
                
                # 添加到提取日志
                log_entry = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tool": "Parameter Recorder",
                    "success": True,
                    "action_id": new_action_id,
                    "details": f"{parameter_name} = {parameter_value} {parameter_unit or ''}"
                }
                if "extraction_log" not in st.session_state:
                    st.session_state.extraction_log = []
                st.session_state.extraction_log.append(log_entry)
                
                st.success(f"Action ID: {new_action_id}")
                
                # 清除会话摘要缓存以强制刷新
                if "session_summary" in st.session_state:
                    del st.session_state["session_summary"]
            else:
                error_msg = result.get("error", "Unknown error") if result else "API call failed"
                st.error(f"❌ Failed to record parameter: {error_msg}")

def render_calculation_tool():
    """渲染计算记录工具 - 两步工作流"""
    st.markdown("### Calculation Recorder")
    st.caption("📐 Pure mathematical operations - No context required")
    
    # 三工具架构说明（简化）
    st.caption("Use **Parameter Recorder** → **Calculation** → **Process Flow**")
    
    # 步骤指示器
    if "verified_calculation" not in st.session_state:
        st.markdown("**Step 1 of 2:** Define and Execute Calculation")
    else:
        st.markdown("**Step 2 of 2:** Verify and Record Calculation")
    
    # 第一步：定义和执行计算
    if "verified_calculation" not in st.session_state:
        # 自动刷新session summary以获取最新的parameters（用于Expression Builder）
        if st.session_state.workbench_session_id:
            # 每次进入Step 1都刷新，确保显示最新的parameters（专家工作台使用 json 格式 + workbench 视图）
            result = call_api(f"/tools/session-summary/{st.session_state.workbench_session_id}?format=json&view=workbench", method="GET")
            if result and result.get("success"):
                st.session_state.session_summary = result["data"]
        
        # 获取已记录的parameters用于辅助构建表达式
        available_params = []
        if st.session_state.get("session_summary"):
            summary_data = st.session_state.session_summary.get("data", {}).get("data", {})
            param_analysis = summary_data.get("parameter_analysis", {})
            parameters = param_analysis.get("parameters", [])
            available_params = parameters
        
        # 计算表达式输入
        st.markdown("#### Expression Builder")
        
        # 如果有已记录的parameters，显示辅助选择器
        if available_params:
            with st.expander("Build from Parameters", expanded=False):
                st.caption("Select parameters to help build your expression")
                
                # 显示可用的parameters
                for p in available_params:
                    param_display = f"**{p['parameter_name']}** = {p['parameter_value']} {p['parameter_unit']} (ID: {p['action_id']})"
                    st.text(param_display)
                
                st.caption("Example: If you select 'power' and 'time', you might write: `power * time`")
        
        calculation_expression = st.text_input(
            "Calculation Expression *",
            placeholder="e.g., power * time, (mass * distance) / capacity",
            help="Enter the mathematical expression. Use parameter names (lowercase) as variables.",
            key="calc_expression_input"
        )
        
        # 变量输入（可选，但推荐从parameters自动生成）
        with st.expander("⚙️ Variables (Auto-fill from Parameters)", expanded=False):
            st.caption("Define variable values. You can copy values from recorded parameters above.")
            variables_text = st.text_area(
                "Variables (JSON format)",
                placeholder='{"power": 10.5, "time": 2.0}',
                help="Enter variables in JSON format. Variable names should match those in your expression.",
                key="calc_variables_input"
            )
        
        # 单位输入
        calculation_unit = st.text_input(
            "Expected Result Unit (Optional)",
            placeholder="e.g., kWh, kg CO2-eq, MJ",
            help="Specify the expected unit of the calculation result",
            key="calc_unit_input"
        )
        
        # 参数预览
        if available_params:
            with st.expander("Session Summary", expanded=False):
                st.caption("Review recorded parameters:")
                for p in available_params:
                    st.text(f"✓ {p['parameter_name']} = {p['parameter_value']} {p['parameter_unit']}")
        
        # Link to previous action
        link_to_action = st.text_input(
            "Link to Previous Action (Optional)",
            value=st.session_state.get("last_action_id", ""),
            placeholder="ACT_0001",
            help="Link this calculation to a previous action",
            key="calc_link_input"
        )
        
        # Calculate & Verify 按钮
        if st.button("Calculate & Verify", use_container_width=True):
            # 验证必填字段
            if not calculation_expression or not calculation_expression.strip():
                st.error("Calculation expression is required")
                return
            
            # 解析变量
            variables = None
            if variables_text and variables_text.strip():
                try:
                    import json
                    variables = json.loads(variables_text.strip())
                    if not isinstance(variables, dict):
                        st.error("Variables must be a JSON object")
                        return
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for variables")
                    return
            
            # 调用执行计算API
            with st.spinner("Executing calculation..."):
                execute_data = {
                    "session_id": st.session_state.workbench_session_id,
                    "expression": calculation_expression.strip(),
                    "variables": variables
                }
                
                result = call_api("/tools/execute-calculation", method="POST", data=execute_data)
                
                if result and result.get("success"):
                    api_response = result.get("data", {})
                    if api_response.get("success"):
                        # 保存验证结果到session state
                        st.session_state["verified_calculation"] = {
                            "expression": calculation_expression.strip(),
                            "result": api_response.get("result"),
                            "unit": calculation_unit.strip() if calculation_unit else None,
                            "link_to": link_to_action.strip() if link_to_action else None,
                            "variables": variables
                        }
                        st.rerun()
                    else:
                        st.error(f"Calculation failed: {api_response.get('error', 'Unknown error')}")
                else:
                    error_msg = result.get("error", "API call failed") if result else "API call failed"
                    st.error(f"Failed to execute calculation: {error_msg}")
    
    # 第二步：验证和记录计算
    else:
        verified_calc = st.session_state["verified_calculation"]
        
        # 自动刷新session summary以获取最新的parameters
        if st.session_state.workbench_session_id and not st.session_state.get("session_summary"):
            result = call_api(f"/tools/session-summary/{st.session_state.workbench_session_id}?format=json&view=workbench", method="GET")
            if result["success"]:
                st.session_state.session_summary = result["data"]
        
        st.markdown("#### Calculation Result")
        
        # 显示计算结果
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text_input(
                "Expression",
                value=verified_calc["expression"],
                disabled=True
            )
        with col2:
            result_display = f"{verified_calc['result']}"
            if verified_calc["unit"]:
                result_display += f" {verified_calc['unit']}"
            st.text_input(
                "Calculated Result",
                value=result_display,
                disabled=True
            )
        
        # 显示其他信息
        if verified_calc["variables"]:
            st.text_area(
                "Variables Used",
                value=str(verified_calc["variables"]),
                disabled=True,
                height=60
            )
        
        if verified_calc["link_to"]:
            st.text_input(
                "Linked to",
                value=verified_calc["link_to"],
                disabled=True
            )
        
        # Data Dependencies选择
        st.markdown("### Data Dependencies")
        
        # 获取当前session的所有parameters
        selected_dependencies = []
        if st.session_state.session_summary:
            summary_data = st.session_state.session_summary.get("data", {}).get("data", {})
            param_analysis = summary_data.get("parameter_analysis", {})
            parameters = param_analysis.get("parameters", [])
            
            if parameters:
                # 构建选项列表
                param_options = []
                for p in parameters:
                    option = f"{p['action_id']} - {p['parameter_name']} = {p['parameter_value']} {p['parameter_unit']}"
                    param_options.append(option)
                
                # 多选框
                selected_options = st.multiselect(
                    "Select parameters used in this calculation",
                    options=param_options
                )
                
                # 提取action_id
                selected_dependencies = [opt.split(" - ")[0] for opt in selected_options]
            else:
                st.info("No parameters recorded yet")
        
        # 操作按钮
        st.markdown("---")
        if st.button("Record Verified Calculation", use_container_width=True):
            # 构建记录请求数据
            calculation_data = {
                "session_id": st.session_state.workbench_session_id,
                "calculation_expression": verified_calc["expression"],
                "calculation_result": verified_calc["result"],
                "calculation_unit": verified_calc["unit"],
                "data_dependencies": selected_dependencies,
                "link_to": verified_calc["link_to"],
                "intent": "calculate"
            }
            
            # 调用记录API
            with st.spinner("Recording calculation..."):
                result = call_api("/tools/record-calculation", method="POST", data=calculation_data)
                
                if result and result.get("success"):
                    st.success("Calculation recorded successfully!")
                    
                    # 更新session state - 修复API响应解析路径
                    api_response = result.get("data", {})
                    new_action_id = api_response.get("data", {}).get("new_action_id") if api_response else None
                    
                    if new_action_id:
                        st.session_state["last_action_id"] = new_action_id
                        st.session_state["last_intent"] = "calculate"
                    
                    # 添加到提取日志
                    log_entry = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "tool": "Calculation Recorder",
                        "success": True,
                        "details": f"Expression: {verified_calc['expression']} = {verified_calc['result']}"
                    }
                    st.session_state.extraction_log.append(log_entry)
                    
                    # 清除会话摘要以强制刷新
                    if "session_summary" in st.session_state:
                        del st.session_state["session_summary"]
                    
                    # 清除验证状态
                    del st.session_state["verified_calculation"]
                    
                    # 显示成功信息和下一步提示
                    st.info(f"Action ID: {new_action_id}")
                    if verified_calc["link_to"]:
                        st.info(f"Linked to: {verified_calc['link_to']}")
                    
                    st.markdown("---")
                    st.markdown("### Next Step: Record Calculation Result")
                    result_unit = verified_calc['unit'] or ''
                    st.info(f"""
                    **Two-Step Workflow Reminder:**
                    1. **Calculation Process Recorded** (Current step completed)
                    2. ⏭️ **Next**: Switch to "Process Flow (Materials & Emissions)" tool to record the calculation result in no-context mode
                    
                    **Instructions for Step 2:**
                    - Use the "Process Flow" tool
                    - Record the calculation result (e.g., "Energy Consumption", value: {verified_calc['result']}, unit: "{result_unit}")
                    - The system will automatically link it to this calculation action
                    """)
                    
                else:
                    error_msg = result.get("error", "Unknown error") if result else "API call failed"
                    st.error(f"Failed to record calculation: {error_msg}")
        
        # Start Over按钮（移到下方）
        st.markdown("")  # 添加一些间距
        if st.button("Start Over", use_container_width=True):
            del st.session_state["verified_calculation"]
            st.rerun()

# ============================================================================
# 主界面
# ============================================================================

def main():
    """主界面函数"""
    init_workbench_state()
    
    # 页面标题
    st.title("EcoLLM Annotation Workbench")
    
    # 三栏式布局
    col_a, col_b, col_c = st.columns([1, 1.5, 1], gap="medium")
    
    with col_a:
        column_a_session_management()
    
    with col_b:
        column_b_document_context()
    
    with col_c:
        column_c_extraction_tools()

if __name__ == "__main__":
    main()
