"""
openLCA IPC 客户端

通过 IPC 协议与 openLCA 软件通信，实现：
1. 导入匹配后的 LCI 数据
2. 执行 LCIA 计算
3. 导出结果
"""

import logging
import requests
import json
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)


class OpenLCAClient:
    """openLCA IPC 客户端"""
    
    def __init__(self, host: str = None, port: int = None):
        """
        初始化客户端
        
        Args:
            host: openLCA IPC 服务地址（默认从环境变量或 localhost）
            port: openLCA IPC 服务端口（默认从环境变量或 8080）
        """
        self.host = host or os.getenv("OPENLCA_HOST", "localhost")
        # 默认端口 8081，避免与 vLLM (8080) 冲突
        self.port = port or int(os.getenv("OPENLCA_PORT", "8081"))
        self.base_url = f"http://{self.host}:{self.port}"
        self._connected = False
        
    def test_connection(self) -> Dict[str, Any]:
        """测试与 openLCA 的连接"""
        try:
            # openLCA IPC 使用 JSON-RPC 2.0
            response = self._call_rpc("data/get/descriptors", {"@type": "Flow"})
            if response.get("error"):
                return {
                    "success": False,
                    "error": response["error"].get("message", "Unknown error")
                }
            self._connected = True
            return {
                "success": True,
                "message": f"Connected to openLCA at {self.base_url}",
                "flow_count": len(response.get("result", []))
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Cannot connect to openLCA at {self.base_url}. Is openLCA running with IPC enabled?"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _call_rpc(self, method: str, params: Dict = None) -> Dict:
        """调用 JSON-RPC 方法"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        response = requests.post(
            self.base_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        return response.json()
    
    def get_flows(self, limit: int = 100) -> Dict[str, Any]:
        """获取 openLCA 中的流列表"""
        try:
            response = self._call_rpc("data/get/descriptors", {"@type": "Flow"})
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            flows = response.get("result", [])[:limit]
            return {
                "success": True,
                "total": len(response.get("result", [])),
                "flows": flows
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_processes(self, limit: int = 100) -> Dict[str, Any]:
        """获取 openLCA 中的过程列表"""
        try:
            response = self._call_rpc("data/get/descriptors", {"@type": "Process"})
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            processes = response.get("result", [])[:limit]
            return {
                "success": True,
                "total": len(response.get("result", [])),
                "processes": processes
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_flow_info(self, flow_id: str) -> Dict[str, Any]:
        """
        获取流量的详细信息（包括 flowProperties 和 units）
        
        Args:
            flow_id: Flow UUID
            
        Returns:
            Flow 详细信息
        """
        try:
            response = self._call_rpc("data/get", {"@type": "Flow", "@id": flow_id})
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "flow": response.get("result")
            }
        except Exception as e:
            logger.error(f"获取 Flow 信息失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_impact_methods(self) -> Dict[str, Any]:
        """获取可用的影响评价方法"""
        try:
            response = self._call_rpc("data/get/descriptors", {"@type": "ImpactMethod"})
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "methods": response.get("result", [])
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_product_system(self, 
                              process_id: str,
                              name: str = None) -> Dict[str, Any]:
        """
        创建产品系统
        
        Args:
            process_id: 参考过程的 UUID
            name: 产品系统名称
        """
        try:
            params = {
                "process": {"@type": "Process", "@id": process_id},
                "name": name or f"Product System - {process_id[:8]}"
            }
            
            response = self._call_rpc("data/create/product_system", params)
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "product_system": response.get("result")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def calculate(self,
                  target_id: str,
                  target_type: str = "Process",
                  impact_method_id: str = None) -> Dict[str, Any]:
        """
        Execute LCA calculation
        
        Args:
            target_id: Target UUID (Process or ProductSystem)
            target_type: Target type ("Process" or "ProductSystem")
            impact_method_id: Impact method UUID (optional)
        """
        try:
            params = {
                "target": {"@type": target_type, "@id": target_id}
            }
            
            if impact_method_id:
                params["impactMethod"] = {"@type": "ImpactMethod", "@id": impact_method_id}
            
            response = self._call_rpc("result/calculate", params)
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "result": response.get("result")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_result_state(self, result_id: str) -> Dict[str, Any]:
        """
        Get calculation result state
        
        Args:
            result_id: Result UUID
        """
        try:
            response = self._call_rpc("result/state", {"@id": result_id})
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "state": response.get("result")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_total_impacts(self, result_id: str) -> Dict[str, Any]:
        """
        Get total impact assessment results
        
        Args:
            result_id: Result UUID
        """
        try:
            response = self._call_rpc("result/total-impacts", {"@id": result_id})
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "impacts": response.get("result", [])
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_total_flows(self, result_id: str) -> Dict[str, Any]:
        """
        Get total flow results (LCI inventory)
        
        Args:
            result_id: Result UUID
            
        Returns:
            Dict with success status and flow results
        """
        try:
            response = self._call_rpc("result/total-flows", {"@id": result_id})
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "flows": response.get("result", [])
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_impact_contributions(self, result_id: str, impact_id: str) -> Dict[str, Any]:
        """
        Get flow contributions to a specific impact category
        
        Args:
            result_id: Result UUID
            impact_id: Impact category UUID
            
        Returns:
            Dict with success status and contribution data
        """
        try:
            params = {
                "@id": result_id,
                "impactCategory": {"@id": impact_id}
            }
            response = self._call_rpc("result/impact-contributions", params)
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "contributions": response.get("result", [])
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_flow_contributions(self, result_id: str, flow_id: str) -> Dict[str, Any]:
        """
        Get impact contributions of a specific flow
        
        Args:
            result_id: Result UUID
            flow_id: Flow UUID
            
        Returns:
            Dict with success status and contribution data
        """
        try:
            params = {
                "@id": result_id,
                "flow": {"@id": flow_id}
            }
            response = self._call_rpc("result/flow-contributions", params)
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "contributions": response.get("result", [])
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_tech_flow_contributions(self, result_id: str) -> Dict[str, Any]:
        """
        Get all tech flow (product/process) contributions
        
        Args:
            result_id: Result UUID
            
        Returns:
            Dict with success status and tech flow data
        """
        try:
            response = self._call_rpc("result/total-requirements", {"@id": result_id})
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "tech_flows": response.get("result", [])
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    
    def dispose_result(self, result_id: str) -> Dict[str, Any]:
        """
        Dispose calculation result to free memory
        
        Args:
            result_id: Result UUID
        """
        try:
            response = self._call_rpc("result/dispose", {"@id": result_id})
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def import_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        导入流到 openLCA
        
        Args:
            flow_data: 流数据（JSON-LD 格式）
        """
        try:
            response = self._call_rpc("data/put", flow_data)
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "message": "Flow imported successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def import_process(self, process_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        导入过程到 openLCA
        
        Args:
            process_data: 过程数据（JSON-LD 格式）
        """
        try:
            response = self._call_rpc("data/put", process_data)
            if response.get("error"):
                return {"success": False, "error": response["error"]}
            
            return {
                "success": True,
                "message": "Process imported successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# 全局实例
_client_instance = None

def get_openlca_client() -> OpenLCAClient:
    """获取全局客户端实例"""
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenLCAClient()
    return _client_instance
