import streamlit as st
import requests
import time
from typing import Optional

# Simple page configuration
st.set_page_config(
    page_title="EcoLLM",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS design
st.markdown("""
<style>
    /* Hide default elements but keep sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    
    /* Keep sidebar toggle button visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
    }
    
    /* Main container */
    .main > div {
        padding-top: 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Title styles */
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 400;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        text-align: center;
        font-size: 1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card container */
    .card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        margin-bottom: 1.5rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    
    .status-active { background-color: #27ae60; }
    .status-inactive { background-color: #e74c3c; }
    
    /* Message styles */
    .user-message {
        background: #f8f9fa;
        border-left: 3px solid #3498db;
        padding: 0.8rem;
        margin: 0.8rem 0;
        border-radius: 0 6px 6px 0;
    }
    
    .assistant-message {
        background: #ffffff;
        border-left: 3px solid #2ecc71;
        padding: 0.8rem;
        margin: 0.8rem 0;
        border-radius: 0 6px 6px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Chat container */
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    .chat-container::-webkit-scrollbar {
        width: 4px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 2px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# 后端API配置
# 使用环境变量或默认 localhost（服务器端渲染时会自动解析）
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# 初始化session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'mode' not in st.session_state:
    st.session_state.mode = "ai_chat"  # 🔥 默认使用 AI Chat 模式
if 'llm_session_id' not in st.session_state:
    st.session_state.llm_session_id = None
if 'llm_chat_history' not in st.session_state:
    st.session_state.llm_chat_history = []

def upload_pdf(file) -> Optional[str]:
    """上传PDF文件到后端（使用统一的 /tools/process-document 接口）"""
    try:
        import base64
        
        # 编码文件内容为base64
        file_content = base64.b64encode(file.getvalue()).decode()
        
        # 调用统一的工具接口
        payload = {
            "file_content": file_content,
            "filename": file.name
        }
        response = requests.post(
            f"{BACKEND_URL}/tools/process-document",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                # 从工具响应中提取session_id
                return result.get("data", {}).get("session_id")
            else:
                st.error(f"上传失败: {result.get('error', '未知错误')}")
                return None
        else:
            st.error(f"上传失败: {response.text}")
            return None
    except Exception as e:
        st.error(f"上传过程中出错: {str(e)}")
        return None

def search_lci_data(session_id: str, instruction: str) -> dict:
    """搜索LCI数据"""
    try:
        payload = {
            "session_id": session_id,
            "instruction": instruction
        }
        
        response = requests.post(
            f"{BACKEND_URL}/search-lci-data",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"请求失败: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"请求过程中出错: {str(e)}"
        }

def check_session_status(session_id: str) -> bool:
    """检查会话状态"""
    try:
        response = requests.get(f"{BACKEND_URL}/session/{session_id}/status")
        if response.status_code == 200:
            result = response.json()
            return result.get("exists", False)
        return False
    except:
        return False

def match_ecoinvent_flow(flow_name: str, category: str = None, top_k: int = 5) -> dict:
    """匹配 ecoinvent 流"""
    try:
        payload = {"flow_name": flow_name, "top_k": top_k}
        if category:
            payload["category"] = category
        response = requests.post(
            f"{BACKEND_URL}/ecoinvent/match-flow",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_all_lci_sessions() -> dict:
    """获取所有有 LCI 数据的 session 列表"""
    try:
        response = requests.get(f"{BACKEND_URL}/lcia/sessions")
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_session_lci_data(session_id: str) -> dict:
    """获取会话的 LCI 数据（用于 LCIA 计算）"""
    try:
        response = requests.get(f"{BACKEND_URL}/lcia/session/{session_id}/data")
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_lcia_methods() -> dict:
    """获取可用的 LCIA 方法列表"""
    try:
        response = requests.get(f"{BACKEND_URL}/lcia/methods?limit=50")
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def match_session_flows(session_id: str, use_llm: bool = False) -> dict:
    """批量匹配会话中所有流
    
    Args:
        session_id: 会话 ID
        use_llm: 是否使用 LLM 辅助重写流名称以提高匹配精度
    """
    try:
        params = {"use_llm": str(use_llm).lower()}  # 明确传递 true/false
        response = requests.get(f"{BACKEND_URL}/lcia/session/{session_id}/match", params=params)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def confirm_flow_match(session_id: str, action_id: str, ecoinvent_uuid: str) -> dict:
    """确认流匹配"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/ecoinvent/confirm-match",
            json={"session_id": session_id, "action_id": action_id, "ecoinvent_uuid": ecoinvent_uuid}
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def calculate_lcia(session_id: str, lcia_method_uuid: str, flow_mappings: list = None) -> dict:
    """执行 LCIA 计算"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/lcia/session/{session_id}/calculate",
            json={"lcia_method_uuid": lcia_method_uuid, "flow_mappings": flow_mappings or []}
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_openlca_connection() -> dict:
    """测试 openLCA 连接"""
    try:
        response = requests.get(f"{BACKEND_URL}/openlca/test", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def configure_openlca(host: str, port: int) -> dict:
    """配置 openLCA IPC 地址"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/openlca/configure",
            json={"host": host, "port": port},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_llm_chat_session(pdf_session_id: str = None) -> Optional[str]:
    """创建LLM聊天会话"""
    try:
        payload = {"pdf_session_id": pdf_session_id}
        response = requests.post(
            f"{BACKEND_URL}/chat/create-session",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("session_id")
        return None
    except Exception as e:
        st.error(f"Failed to create chat session: {str(e)}")
        return None

def send_llm_message(llm_session_id: str, message: str, pdf_session_id: str = None) -> dict:
    """发送消息给LLM"""
    try:
        payload = {
            "session_id": llm_session_id,
            "message": message
        }
        
        if pdf_session_id:
            payload["message"] = f"[PDF_SESSION_ID: {pdf_session_id}] {message}"
        
        response = requests.post(
            f"{BACKEND_URL}/chat/message",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"请求失败: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"请求过程中出错: {str(e)}"
        }

def send_llm_message_stream(llm_session_id: str, message: str, pdf_session_id: str = None):
    """
    流式发送消息给LLM
    
    Yields:
        Dict with type: "thinking", "content", "tool_call", "done", "error"
    """
    import json
    
    try:
        payload = {
            "session_id": llm_session_id,
            "message": message
        }
        
        if pdf_session_id:
            payload["message"] = f"[PDF_SESSION_ID: {pdf_session_id}] {message}"
        
        # 使用流式请求
        response = requests.post(
            f"{BACKEND_URL}/chat/stream",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if response.status_code != 200:
            yield {"type": "error", "error": f"请求失败: {response.status_code}"}
            return
        
        # 解析 SSE 流
        buffer = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    try:
                        parsed = json.loads(data)
                        parsed["type"] = event_type if event_type else parsed.get("type", "content")
                        yield parsed
                    except json.JSONDecodeError:
                        pass
                    event_type = None
                    
    except Exception as e:
        yield {"type": "error", "error": str(e)}

# ==================== SIDEBAR: LCIA Calculation ====================

# 初始化侧边栏状态
if "lci_data" not in st.session_state:
    st.session_state.lci_data = None
if "match_results" not in st.session_state:
    st.session_state.match_results = None
if "selected_method" not in st.session_state:
    st.session_state.selected_method = None
if "available_sessions" not in st.session_state:
    st.session_state.available_sessions = []
if "selected_lcia_session" not in st.session_state:
    st.session_state.selected_lcia_session = None

with st.sidebar:
    st.markdown("### LCIA Calculation")
    
    # Session 选择
    with st.expander("Session", expanded=True):
        # 刷新按钮和当前 session 显示
        if st.button("Refresh Sessions", use_container_width=True, key="refresh_sessions"):
            result = get_all_lci_sessions()
            if result.get("success"):
                st.session_state.available_sessions = result.get("sessions", [])
                if st.session_state.available_sessions:
                    st.success(f"Found {len(st.session_state.available_sessions)} sessions")
                else:
                    st.info("No LCI data in database")
        
        # 显示可用的 session
        if st.session_state.available_sessions:
            session_options = [
                f"{s['session_id'][:8]}...{s['session_id'][-4:]} ({s['flow_count']} flows)"
                for s in st.session_state.available_sessions
            ]
            selected_idx = st.selectbox(
                "Select session",
                options=range(len(session_options)),
                format_func=lambda i: session_options[i],
                key="session_selector"
            )
            st.session_state.selected_lcia_session = st.session_state.available_sessions[selected_idx]["session_id"]
        elif st.session_state.session_id:
            # 使用当前 session
            st.session_state.selected_lcia_session = st.session_state.session_id
        
        # 显示当前选中的 session
        if st.session_state.selected_lcia_session:
            st.caption("Current session:")
            st.code(st.session_state.selected_lcia_session, language=None)
    
    # Load LCI Data
    with st.expander("Load LCI Data", expanded=False):
        if st.session_state.selected_lcia_session:
            if st.button("Load Data", use_container_width=True, key="load_lci"):
                with st.spinner("Loading..."):
                    result = get_session_lci_data(st.session_state.selected_lcia_session)
                    if result.get("success"):
                        st.session_state.lci_data = result
                        st.success(f"Loaded {result.get('total_flows', 0)} flows")
                    else:
                        st.error(result.get("error", "Failed")[:50])
            
            # 显示已加载的数据
            if st.session_state.lci_data:
                data = st.session_state.lci_data
                st.markdown(f"**Inputs:** {len(data.get('inputs', []))}")
                for f in data.get("inputs", [])[:3]:
                    st.caption(f"• {f['name'][:25]}... ({f['value']} {f['unit']})")
                st.markdown(f"**Outputs:** {len(data.get('outputs', []))}")
                for f in data.get("outputs", [])[:2]:
                    st.caption(f"• {f['name'][:25]}... ({f['value']} {f['unit']})")
        else:
            st.info("Select a session first")
    
    # Match Ecoinvent
    with st.expander("Match Ecoinvent", expanded=False):
        if st.session_state.lci_data:
            # LLM 辅助匹配选项
            col1, col2 = st.columns([3, 1])
            with col1:
                use_llm = st.checkbox("Use LLM-assisted matching", value=False, 
                                      help="Use LLM to rewrite flow names for better matching accuracy (slower but more accurate)")
            with col2:
                if st.session_state.match_results:
                    matched = sum(1 for r in st.session_state.match_results.get("results", []) if r.get("matches"))
                    total = st.session_state.match_results.get('total_flows', 0)
                    st.metric("Matched", f"{matched}/{total}")
            
            if st.button("Auto Match All", use_container_width=True, key="match_all"):
                # 清除旧的匹配结果
                st.session_state.match_results = None
                
                with st.spinner("Matching flows to ecoinvent..." + (" (with LLM)" if use_llm else " (with context)")):
                    # 使用批量匹配 API，自动传递上下文信息
                    result = match_session_flows(st.session_state.selected_lcia_session, use_llm=use_llm)
                    if result.get("success"):
                        st.session_state.match_results = result
                        matched = sum(1 for r in result.get("results", []) if r.get("matches"))
                        total = result.get('total_flows', 0)
                        avg_sim = sum(r.get("matches", [{}])[0].get("similarity", 0) 
                                     for r in result.get("results", []) if r.get("matches")) / max(matched, 1)
                        st.success(f"✓ Matched {matched}/{total} flows (avg similarity: {avg_sim:.2f})")
                        st.rerun()  # 强制刷新显示
                    else:
                        st.error(result.get("error", "Failed")[:50])
            
            # 显示匹配结果
            if st.session_state.match_results:
                st.markdown("---")
                
                for r in st.session_state.match_results.get("results", []):
                    orig = r.get("original", {})
                    matches = r.get("matches", [])
                    skip_reason = r.get("skip_reason")
                    
                    st.markdown(f"**{orig.get('name', 'N/A')}**")
                    st.caption(f"{orig.get('value', '')} {orig.get('unit', '')} | {orig.get('category', '')}")
                    
                    if skip_reason:
                        st.text(f"  Skipped: {skip_reason}")
                    elif matches:
                        best = matches[0]
                        st.text(f"  → {best['name']}")
                    else:
                        st.text("  → No match found")
                    st.markdown("---")
        else:
            st.info("Load LCI data first")
    
    # Select LCIA Method
    with st.expander("Select LCIA Method", expanded=False):
        if st.button("Load Methods", use_container_width=True, key="load_methods"):
            with st.spinner("Loading..."):
                result = get_lcia_methods()
                if result.get("success"):
                    st.session_state.lcia_methods = result.get("methods", [])
                    st.success(f"Loaded {len(st.session_state.lcia_methods)} methods")
        
        if "lcia_methods" in st.session_state and st.session_state.lcia_methods:
            method_names = [m.get("name", "Unknown")[:50] for m in st.session_state.lcia_methods]
            selected_idx = st.selectbox("Method", range(len(method_names)), 
                                        format_func=lambda i: method_names[i],
                                        key="method_select")
            st.session_state.selected_method = st.session_state.lcia_methods[selected_idx]
            st.caption(f"UUID: {st.session_state.selected_method.get('uuid', 'N/A')[:20]}...")
    
    # Calculate LCIA
    with st.expander("Calculate LCIA", expanded=False):
        # openLCA 连接测试
        if st.button("Test openLCA Connection", use_container_width=True, key="test_conn"):
            result = test_openlca_connection()
            if result.get("success"):
                st.success("✓ openLCA IPC connected")
            else:
                st.error(f"✗ {result.get('error', 'Connection failed')[:40]}")
        
        st.markdown("---")
        
        # 计算按钮
        can_calculate = (st.session_state.lci_data and 
                        st.session_state.match_results and 
                        st.session_state.selected_method)
        
        if st.button("Run LCIA Calculation", use_container_width=True, 
                    disabled=not can_calculate, key="run_lcia"):
            with st.spinner("Calculating LCIA..."):
                result = calculate_lcia(
                    st.session_state.selected_lcia_session,
                    st.session_state.selected_method.get("uuid"),
                    []
                )
                
                st.session_state.lcia_result = result
        
        if not can_calculate:
            missing = []
            if not st.session_state.lci_data:
                missing.append("LCI data")
            if not st.session_state.match_results:
                missing.append("Ecoinvent match")
            if not st.session_state.selected_method:
                missing.append("LCIA method")
            st.caption(f"Missing: {', '.join(missing)}")
    
    # Display LCIA results in sidebar
    if hasattr(st.session_state, 'lcia_result') and st.session_state.lcia_result:
        result = st.session_state.lcia_result
        if result.get("success"):
            results = result.get("results", {})
            status = results.get("status")
            
            if status == "completed":
                # Display completed LCIA calculation results
                st.success("✓ LCIA Calculation Completed!")
                st.info(results.get("message", "LCIA calculation completed successfully"))
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Flows", results.get("total_flows", 0))
                with col2:
                    st.metric("Matched", results.get("exchanges_count", 0))
                with col3:
                    st.metric("Impact Categories", results.get("impact_count", 0))
                
                # Display functional unit
                st.markdown(f"**Functional Unit:** {results.get('functional_unit', 'N/A')}")
                
                # Display impact results
                st.markdown("### Environmental Impact Results")
                impacts = results.get("impacts", [])
                if impacts:
                    for impact in impacts:
                        with st.expander(f"{impact.get('category', 'Unknown')}: {impact.get('amount', 0):.6e} {impact.get('unit', '')}"):
                            st.write(f"**Amount:** {impact.get('amount', 0):.6e}")
                            st.write(f"**Unit:** {impact.get('unit', 'N/A')}")
                            if impact.get('description'):
                                st.write(f"**Description:** {impact.get('description')}")
                else:
                    st.warning("No impact results available")
                
                # Display major flows inventory
                flow_contributions = results.get("flow_contributions", [])
                
                if flow_contributions:
                    st.markdown("### Major Flows Inventory")
                    st.caption(f"Showing top {len(flow_contributions)} flows by amount")
                    
                    for i, flow_contrib in enumerate(flow_contributions, 1):
                        flow_name = flow_contrib.get("flow_name", "Unknown")
                        flow_category = flow_contrib.get("flow_category", "")
                        flow_amount = flow_contrib.get("flow_amount", 0)
                        flow_unit = flow_contrib.get("flow_unit", "")
                        is_input = flow_contrib.get("is_input", False)
                        
                        # Determine flow direction
                        direction = "Input" if is_input else "Output"
                        
                        # Create expander title with flow info
                        expander_title = f"{i}. {flow_name}: {flow_amount:.4g} {flow_unit} ({direction})"
                        
                        with st.expander(expander_title):
                            st.write(f"**Flow:** {flow_name}")
                            st.write(f"**Category:** {flow_category if flow_category else 'N/A'}")
                            st.write(f"**Amount:** {flow_amount:.6e} {flow_unit}")
                            st.write(f"**Direction:** {direction}")
                            
                            # Add explanation
                            if abs(flow_amount) > 1000:
                                st.info("This is a major contributor to the inventory")
                            elif abs(flow_amount) < 0:
                                st.info("Negative value indicates avoided burden or credit")
            
            elif status == "ready":
                # Display prepared data (fallback)
                st.success("✓ LCIA Data Prepared!")
                st.info(results.get("message", "LCIA calculation data has been prepared"))
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Flows", results.get("total_flows", 0))
                with col2:
                    st.metric("Matched", results.get("exchanges_count", 0))
                with col3:
                    functional_unit = results.get("functional_unit", "N/A")
                    if len(str(functional_unit)) > 20:
                        functional_unit = str(functional_unit)[:17] + "..."
                    st.metric("Functional Unit", functional_unit)
                
                # Display exchanges list
                st.markdown("### 📊 Exchanges Data")
                exchanges = results.get("exchanges", [])
                if exchanges:
                    for i, ex in enumerate(exchanges, 1):
                        with st.expander(f"Exchange {i}: {ex.get('amount')} {ex.get('unit')}"):
                            st.json(ex)
                
                # Display note
                if result.get("note"):
                    st.warning(result.get("note"))
            else:
                st.success("✓ Calculation complete!")
                st.json(results)
        else:
            st.error(result.get("error", "Failed"))

# ==================== MAIN INTERFACE ====================

# Title section
st.markdown('<h1 class="main-title">EcoLLM</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Life Cycle Assessment Analysis Platform</p>', unsafe_allow_html=True)

# Status and reset
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    if st.session_state.session_id and check_session_status(st.session_state.session_id):
        st.markdown('<span class="status-indicator status-active"></span>Document Ready', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-inactive"></span>No Document', unsafe_allow_html=True)
        if st.session_state.session_id:
            st.session_state.session_id = None

with col2:
    if st.session_state.llm_session_id:
        st.markdown('<span class="status-indicator status-active"></span>AI Assistant Ready', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-inactive"></span>AI Assistant Inactive', unsafe_allow_html=True)

with col3:
    if st.button("Reset", type="secondary", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.session_state.llm_session_id = None
        st.session_state.llm_chat_history = []
        if hasattr(st.session_state, 'start_mode'):
            del st.session_state.start_mode
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Starting options
if not st.session_state.session_id and not st.session_state.llm_session_id:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Choose Analysis Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Document Analysis**")
        st.markdown("Upload PDF for document-based analysis")
        if st.button("Upload Document", type="primary", use_container_width=True):
            st.session_state.start_mode = "document"
            st.rerun()
    
    with col2:
        st.markdown("**LCA Assistant**")
        st.markdown("Chat with AI assistant (no document needed)")
        if st.button("Start Chat", type="secondary", use_container_width=True):
            with st.spinner("Initializing..."):
                llm_session_id = create_llm_chat_session(pdf_session_id=None)
                if llm_session_id:
                    st.session_state.llm_session_id = llm_session_id
                    st.session_state.start_mode = "standalone"
                    st.success("AI assistant ready")
                    st.rerun()
                else:
                    st.error("Failed to initialize AI assistant")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Document upload
if hasattr(st.session_state, 'start_mode') and st.session_state.start_mode == "document" and not st.session_state.session_id:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Upload Document")
    
    uploaded_file = st.file_uploader(
        label="Choose PDF file",
        type=['pdf'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # 检查是否已经处理过这个文件（避免重复处理）
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            with st.spinner("Processing document..."):
                session_id = upload_pdf(uploaded_file)
                
                if session_id:
                    st.session_state.session_id = session_id
                    st.session_state.last_uploaded_file = uploaded_file.name
                    # 不立即 rerun，让成功消息显示
                else:
                    st.error("❌ Failed to process document. Please check backend logs.")
        
        # 显示处理结果（rerun 后也能看到）
        if st.session_state.session_id:
            st.success(f"✅ Document processed successfully!")
            st.info(f"📄 Session ID: `{st.session_state.session_id[:8]}...`")
            
            # 添加按钮进入分析模式
            if st.button("Continue to Analysis →", type="primary", use_container_width=True):
                st.rerun()
    
    if st.button("← Back"):
        if hasattr(st.session_state, 'start_mode'):
            del st.session_state.start_mode
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis mode selection (for document-based)
if st.session_state.session_id:
    # 🔥 自动初始化 AI Chat（不再需要手动切换模式）
    if not st.session_state.llm_session_id:
        with st.spinner("Initializing AI Chat..."):
            llm_session_id = create_llm_chat_session(st.session_state.session_id)
            if llm_session_id:
                st.session_state.llm_session_id = llm_session_id
                st.session_state.mode = "ai_chat"
                # 添加初始欢迎消息到聊天历史
                if not st.session_state.llm_chat_history:
                    session_id_short = st.session_state.session_id[:8] if st.session_state.session_id else "unknown"
                    st.session_state.llm_chat_history.append({
                        "role": "assistant",
                        "content": f"✅ Document successfully processed and ready for analysis!\n\nDocument ID: {session_id_short}...\n\nI have access to the document content and can search through it to answer your questions. What would you like to know about the document?"
                    })
                st.rerun()
            else:
                st.error("Failed to initialize AI assistant")
    
# Analysis interface
if st.session_state.session_id and st.session_state.mode:
    
    # AI Chat Mode (document-based)
    if st.session_state.mode == "ai_chat":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### AI Chat")
        
        # Chat history
        if st.session_state.llm_chat_history:
            for i, msg in enumerate(st.session_state.llm_chat_history):
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        # 🔥 新增：如果有思考过程，用折叠面板显示
                        if msg.get("thinking"):
                            with st.expander("💭 Thinking Process", expanded=False):
                                st.text(msg["thinking"])
                        
                        # 显示实际回复
                        st.write(msg["content"])
                        
                        # Tool results
                        if msg.get("tool_results"):
                            with st.expander(f"🔧 Tool Results", expanded=False):
                                for tool_result in msg["tool_results"]:
                                    st.markdown(f"**{tool_result.get('tool_name', 'Unknown')}**")
                                    if tool_result.get("success"):
                                        st.success("Success")
                                        if tool_result.get("result"):
                                            st.json(tool_result["result"])
                                    else:
                                        st.error(f"Error: {tool_result.get('error', 'Unknown')}")
        
        # Chat input
        user_input = st.chat_input("Ask about your document...")
        
        if user_input and st.session_state.llm_session_id:
            st.session_state.llm_chat_history.append({
                "role": "user", 
                "content": user_input
            })
            
            # Tokens-level streaming mode (Snapshot-based)
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                thinking_container = st.empty()
                content_placeholder = st.empty()
                
                # 🔥 显示初始加载状态
                status_placeholder.markdown("Analyzing...")
                
                thinking_text = ""
                content_text = ""
                tool_results = []
                tool_calls_seen = []
                first_response_received = False
                
                try:
                    for chunk in send_llm_message_stream(
                        llm_session_id=st.session_state.llm_session_id,
                        message=user_input,
                        pdf_session_id=st.session_state.session_id
                    ):
                        chunk_type = chunk.get("type", "")
                        
                        if chunk_type == "thinking":
                            # 收到第一个响应，清除加载状态
                            if not first_response_received:
                                status_placeholder.empty()
                                first_response_received = True
                            # Snapshot mode: replace entire thinking content
                            thinking_text = chunk.get("content", "")
                            with thinking_container.container():
                                with st.expander("Thinking...", expanded=True):
                                    st.markdown(thinking_text)
                        
                        elif chunk_type == "tool_call":
                            tool_name = chunk.get("tool_name", "")
                            tool_calls_seen.append(tool_name)
                            status_placeholder.info(f"Calling tool: {tool_name}...")
                        
                        elif chunk_type == "tool_result":
                            tool_name = chunk.get("tool_name", "")
                            tool_results.append(chunk)
                            status_placeholder.success(f"Tool {tool_name} completed")
                        
                        elif chunk_type == "content":
                            # Snapshot mode: replace entire content
                            content_text = chunk.get("content", "")
                            status_placeholder.empty()
                            # Collapse thinking when content starts
                            if thinking_text:
                                with thinking_container.container():
                                    with st.expander("Thinking Process", expanded=False):
                                        st.markdown(thinking_text)
                            content_placeholder.markdown(content_text)
                        
                        elif chunk_type == "done":
                            status_placeholder.empty()
                            if not content_text:
                                content_text = chunk.get("content", "")
                            if not thinking_text:
                                thinking_text = chunk.get("thinking", "")
                            # Final display
                            if thinking_text:
                                with thinking_container.container():
                                    with st.expander("Thinking Process", expanded=False):
                                        st.markdown(thinking_text)
                            if content_text:
                                content_placeholder.markdown(content_text)
                            elif tool_results:
                                content_text = "I've processed your request. The data has been recorded."
                                content_placeholder.markdown(content_text)
                        
                        elif chunk_type == "error":
                            status_placeholder.error(chunk.get('error', 'Unknown error'))
                            content_text = f"Error: {chunk.get('error', 'Unknown error')}"
                
                except Exception as e:
                    status_placeholder.error(f"Error: {str(e)}")
                    content_text = f"Error: {str(e)}"
                
                # Save to history
                assistant_msg = {
                    "role": "assistant",
                    "content": content_text.strip() if content_text else "Request processed.",
                    "thinking": thinking_text.strip() if thinking_text else None,
                    "tool_results": tool_results if tool_results else None
                }
                st.session_state.llm_chat_history.append(assistant_msg)
            
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Standalone AI Chat Mode
elif st.session_state.llm_session_id and not st.session_state.session_id:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### LCA Assistant")
    
    # Chat history
    if st.session_state.llm_chat_history:
        for i, msg in enumerate(st.session_state.llm_chat_history):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    # 🔥 新增：如果有思考过程，用折叠面板显示
                    if msg.get("thinking"):
                        with st.expander("💭 Thinking Process", expanded=False):
                            st.text(msg["thinking"])
                    
                    # 显示实际回复
                    st.write(msg["content"])
                    
                    # Tool results
                    if msg.get("tool_results"):
                        with st.expander(f"🔧 Tool Results", expanded=False):
                            for tool_result in msg["tool_results"]:
                                st.markdown(f"**{tool_result.get('tool_name', 'Unknown')}**")
                                if tool_result.get("success"):
                                    st.success("Success")
                                    if tool_result.get("result"):
                                        st.json(tool_result["result"])
                                else:
                                    st.error(f"Error: {tool_result.get('error', 'Unknown')}")
    
    # Chat input
    user_input = st.chat_input("Ask about LCA methodology or request data...")
    
    if user_input and st.session_state.llm_session_id:
        st.session_state.llm_chat_history.append({
            "role": "user", 
            "content": user_input
        })
        
        with st.spinner("AI thinking..."):
            result = send_llm_message(
                llm_session_id=st.session_state.llm_session_id, 
                message=user_input,
                pdf_session_id=None
            )
            
            if result and result.get("success"):
                assistant_msg = {
                    "role": "assistant", 
                    "content": result.get("message", "No response"),
                    "thinking": result.get("thinking", ""),  # 🔥 新增：思考过程
                    "tool_results": result.get("tool_results")
                }
                st.session_state.llm_chat_history.append(assistant_msg)
            else:
                error_msg = {
                    "role": "assistant", 
                    "content": f"Error: {result.get('error', 'Unknown error')}"
                }
                st.session_state.llm_chat_history.append(error_msg)
        
        st.rerun()
    
    # Reset option
    if st.button("New Conversation"):
        st.session_state.llm_session_id = None
        st.session_state.llm_chat_history = []
        if hasattr(st.session_state, 'start_mode'):
            del st.session_state.start_mode
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)