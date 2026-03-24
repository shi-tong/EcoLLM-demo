"""
LCIA 计算服务

完整流程：
1. 从 lca_actions 拉取 session 数据
2. 与 ecoinvent flows/processes 匹配
3. 选择 LCIA 方法
4. 生成 olca-ipc 调用参数
5. 通过 IPC 调用 openLCA 计算
6. 返回 LCIA 结果
"""

import logging
from typing import Dict, Any, List, Optional
from .mongodb_manager import mongodb_manager
from .ecoinvent_matcher import get_ecoinvent_matcher
from .openlca_client import get_openlca_client

logger = logging.getLogger(__name__)


class LCIACalculator:
    """LCIA 计算器"""
    
    def __init__(self):
        self.db = None
        self.matcher = None
        self.olca_client = None
        
    def _ensure_initialized(self):
        """确保初始化"""
        if self.db is None:
            self.db = mongodb_manager.get_database()
        if self.matcher is None:
            self.matcher = get_ecoinvent_matcher()
            # 设置 matcher 的 db 连接
            if self.matcher.db is None:
                self.matcher.db = self.db
            self.matcher._ensure_initialized()
        if self.olca_client is None:
            self.olca_client = get_openlca_client()
    
    def get_all_sessions(self) -> Dict[str, Any]:
        """
        获取所有有 LCI 数据的 session 列表
        
        Returns:
            session 列表
        """
        self._ensure_initialized()
        
        try:
            # 获取所有唯一的 session_id
            sessions = self.db.lca_actions.distinct("session_id")
            
            result = []
            for sid in sessions:
                # 获取每个 session 的流数量
                flow_count = self.db.lca_actions.count_documents({
                    "session_id": sid,
                    "record_type": "flow"
                })
                
                # 获取 scope 信息
                scope = self.db.lca_actions.find_one({
                    "session_id": sid,
                    "record_type": "scope"
                })
                
                result.append({
                    "session_id": sid,
                    "flow_count": flow_count,
                    "functional_unit": scope.get("description", "N/A") if scope else "N/A"
                })
            
            return {
                "success": True,
                "sessions": result,
                "total": len(result)
            }
            
        except Exception as e:
            logger.error(f"获取 session 列表失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_session_lci_data(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话的 LCI 数据
        
        Args:
            session_id: 会话 ID
            
        Returns:
            LCI 数据，包括 scope 和 flows
        """
        self._ensure_initialized()
        
        try:
            # 获取 scope 定义
            scope = self.db.lca_actions.find_one({
                "session_id": session_id,
                "record_type": "scope"
            })
            
            # 获取所有 flows
            flows = list(self.db.lca_actions.find({
                "session_id": session_id,
                "record_type": "flow"
            }))
            
            # 按类别分组
            inputs = []
            outputs = []
            
            for flow in flows:
                flow_data = {
                    "action_id": flow.get("action_id"),
                    "name": flow.get("name"),
                    "value": flow.get("value"),
                    "unit": flow.get("unit"),
                    "category": flow.get("category"),
                    "flow_type": flow.get("flow_type"),
                    "ecoinvent_match": flow.get("ecoinvent_match"),
                }
                
                # 根据类别判断输入/输出
                category = flow.get("category", "")
                if category in ["Raw Material", "Process Energy", "Post-processing Energy", 
                               "Feedstock Energy", "Gas", "Cooling Media"]:
                    inputs.append(flow_data)
                else:
                    outputs.append(flow_data)
            
            return {
                "success": True,
                "session_id": session_id,
                "scope": {
                    "functional_unit": scope.get("description") if scope else None,
                    "value": scope.get("value") if scope else None,
                    "unit": scope.get("unit") if scope else None,
                } if scope else None,
                "inputs": inputs,
                "outputs": outputs,
                "total_flows": len(flows)
            }
            
        except Exception as e:
            logger.error(f"获取 LCI 数据失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_lcia_methods(self, limit: int = 50) -> Dict[str, Any]:
        """
        获取可用的 LCIA 方法列表
        
        Returns:
            LCIA 方法列表
        """
        self._ensure_initialized()
        
        try:
            methods = list(self.db.lcia_methods.find(
                {},
                {"uuid": 1, "name": 1, "category": 1, "impact_categories_count": 1, "_id": 0}
            ).limit(limit))
            
            return {
                "success": True,
                "methods": methods,
                "total": len(methods)
            }
            
        except Exception as e:
            logger.error(f"获取 LCIA 方法失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_lcia_method_details(self, method_uuid: str) -> Dict[str, Any]:
        """
        获取 LCIA 方法详情，包括影响类别
        
        Args:
            method_uuid: LCIA 方法 UUID
            
        Returns:
            方法详情
        """
        self._ensure_initialized()
        
        try:
            method = self.db.lcia_methods.find_one(
                {"uuid": method_uuid},
                {"embedding_vector": 0}
            )
            
            if not method:
                return {"success": False, "error": "未找到该 LCIA 方法"}
            
            return {
                "success": True,
                "method": {
                    "uuid": method.get("uuid"),
                    "name": method.get("name"),
                    "category": method.get("category"),
                    "impact_categories": method.get("impactCategories", []),
                }
            }
            
        except Exception as e:
            logger.error(f"获取 LCIA 方法详情失败: {e}")
            return {"success": False, "error": str(e)}
    
    def match_all_flows(self, session_id: str, use_llm_rewrite: bool = False) -> Dict[str, Any]:
        """
        批量匹配会话中所有流
        
        Args:
            session_id: 会话 ID
            use_llm_rewrite: 是否使用 LLM 辅助重写流名称以提高匹配精度
            
        Returns:
            匹配结果
        """
        self._ensure_initialized()
        return self.matcher.batch_match_session(session_id, use_llm_rewrite=use_llm_rewrite)
    
    def prepare_lcia_calculation(self, 
                                  session_id: str,
                                  lcia_method_uuid: str,
                                  flow_mappings: List[Dict] = None,
                                  auto_match: bool = True) -> Dict[str, Any]:
        """
        准备 LCIA 计算参数
        
        Args:
            session_id: 会话 ID
            lcia_method_uuid: LCIA 方法 UUID
            flow_mappings: 流映射列表 [{action_id, ecoinvent_uuid}, ...]
            auto_match: 如果没有映射，是否自动执行匹配（默认 True）
            
        Returns:
            准备好的计算参数
        """
        self._ensure_initialized()
        
        try:
            # 获取 LCI 数据
            lci_data = self.get_session_lci_data(session_id)
            if not lci_data.get("success"):
                return lci_data
            
            # 获取 LCIA 方法
            method = self.db.lcia_methods.find_one({"uuid": lcia_method_uuid})
            if not method:
                return {"success": False, "error": "未找到 LCIA 方法"}
            
            # 如果没有提供映射且启用自动匹配，执行匹配
            # 注意：空列表 [] 也视为没有提供映射
            logger.info(f"检查自动匹配条件: flow_mappings={flow_mappings}, auto_match={auto_match}")
            
            if (not flow_mappings or len(flow_mappings) == 0) and auto_match:
                logger.info(f"触发自动匹配: 未提供流映射，开始执行匹配...")
                match_result = self.match_all_flows(session_id, use_llm_rewrite=False)
                
                logger.info(f"匹配结果: success={match_result.get('success')}, results_count={len(match_result.get('results', []))}")
                
                if not match_result.get("success"):
                    return {
                        "success": False,
                        "error": f"自动匹配失败: {match_result.get('error')}"
                    }
                
                # 从匹配结果构建映射
                flow_mappings = []
                for result in match_result.get("results", []):
                    # action_id 在 result 的顶层，不在 original 中
                    action_id = result.get("action_id")
                    orig = result.get("original", {})
                    matches = result.get("matches", [])
                    if matches and action_id:
                        best_match = matches[0]
                        flow_mappings.append({
                            "action_id": action_id,
                            "ecoinvent_uuid": best_match.get("uuid")
                        })
                
                logger.info(f"自动匹配完成，生成 {len(flow_mappings)} 个映射")
            else:
                logger.info(f"跳过自动匹配: flow_mappings 已提供或 auto_match=False")
            
            # 构建计算参数
            exchanges = []
            all_flows = lci_data.get("inputs", []) + lci_data.get("outputs", [])
            
            for flow in all_flows:
                ecoinvent_uuid = None
                
                # 检查是否有预设映射
                if flow_mappings:
                    for mapping in flow_mappings:
                        if mapping.get("action_id") == flow.get("action_id"):
                            ecoinvent_uuid = mapping.get("ecoinvent_uuid")
                            break
                
                # 如果没有映射，使用已确认的匹配
                if not ecoinvent_uuid and flow.get("ecoinvent_match"):
                    ecoinvent_uuid = flow["ecoinvent_match"].get("uuid")
                
                if ecoinvent_uuid:
                    # 获取匹配的流量信息（包括单位）
                    matched_flow = None
                    if flow.get("ecoinvent_match"):
                        matched_flow = flow["ecoinvent_match"]
                    
                    # 获取原始值和单位
                    original_value = flow.get("value", 0)
                    original_unit = flow.get("unit")
                    matched_unit = matched_flow.get("unit") if matched_flow else None
                    
                    # 如果单位不匹配且兼容，进行转换
                    final_value = original_value
                    final_unit = original_unit
                    
                    if matched_unit and original_unit != matched_unit:
                        from .unit_compatibility import are_units_compatible, convert_unit
                        if are_units_compatible(original_unit, matched_unit):
                            final_value = convert_unit(original_value, original_unit, matched_unit)
                            final_unit = matched_unit
                            logger.info(f"单位转换: {original_value} {original_unit} → {final_value} {matched_unit}")
                    
                    exchanges.append({
                        "flow_uuid": ecoinvent_uuid,
                        "amount": final_value,
                        "unit": final_unit,
                        "original_amount": original_value,
                        "original_unit": original_unit,
                        "matched_unit": matched_unit,
                        "is_input": flow.get("category") in [
                            "Raw Material", "Process Energy", "Post-processing Energy",
                            "Feedstock Energy", "Gas", "Cooling Media"
                        ]
                    })
            
            return {
                "success": True,
                "session_id": session_id,
                "lcia_method": {
                    "uuid": method.get("uuid"),
                    "name": method.get("name"),
                },
                "exchanges": exchanges,
                "total_mapped": len(exchanges),
                "total_flows": len(all_flows),
                "ready_for_calculation": len(exchanges) > 0
            }
            
        except Exception as e:
            logger.error(f"准备 LCIA 计算失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_process_in_openlca(self, 
                                    session_id: str,
                                    exchanges: List[Dict],
                                    functional_unit) -> Dict[str, Any]:
        """
        在 openLCA 中创建临时 Process
        
        Args:
            session_id: 会话 ID
            exchanges: 交换列表（包含 flow_uuid, amount, unit, is_input）
            functional_unit: 功能单位信息（可以是字符串或字典）
            
        Returns:
            创建结果，包含 process_id
        """
        try:
            import uuid
            
            # 构建 openLCA Process JSON-LD 格式
            process_id = str(uuid.uuid4())
            
            # Handle functional_unit as string or dict
            if isinstance(functional_unit, str):
                fu_desc = functional_unit
                fu_value = 1.0
            elif isinstance(functional_unit, dict):
                fu_desc = functional_unit.get("description", "1 unit")
                fu_value = functional_unit.get("value", 1.0)
            else:
                fu_desc = "1 unit"
                fu_value = 1.0
            
            # 构建 exchanges
            olca_exchanges = []
            for i, ex in enumerate(exchanges):
                # 获取 Flow 的详细信息以正确设置单位
                flow_info = self.olca_client.get_flow_info(ex.get("flow_uuid"))
                
                olca_exchange = {
                    "@type": "Exchange",
                    "internalId": i + 1,
                    "flow": {"@type": "Flow", "@id": ex.get("flow_uuid")},
                    "amount": ex.get("amount", 0),
                    "isInput": ex.get("is_input", True),
                    "isQuantitativeReference": False,
                }
                
                # 关键：添加 unit 和 flowProperty（如果可用）
                if flow_info and flow_info.get("success"):
                    flow_data = flow_info.get("flow", {})
                    # 使用第一个 flow property（通常是参考属性）
                    if flow_data.get("flowProperties"):
                        flow_prop = flow_data["flowProperties"][0]
                        olca_exchange["flowProperty"] = {
                            "@type": "FlowProperty",
                            "@id": flow_prop.get("flowProperty", {}).get("@id")
                        }
                        # 使用参考单位
                        if flow_prop.get("referenceUnit"):
                            olca_exchange["unit"] = {
                                "@type": "Unit",
                                "@id": flow_prop["referenceUnit"].get("@id")
                            }
                
                olca_exchanges.append(olca_exchange)
            
            # 添加一个参考产品流（功能单位）
            # 使用第一个 Product 类型的流作为参考流，如果没有则使用第一个输出流
            ref_flow_id = None
            
            # 尝试从 exchanges 中找到一个合适的参考流
            for ex in exchanges:
                if not ex.get("is_input", True):  # 找输出流
                    ref_flow_id = ex.get("flow_uuid")
                    break
            
            # 如果没有找到输出流，使用一个通用的产品流 UUID（需要在 openLCA 中存在）
            # 或者跳过参考流，让 openLCA 使用第一个输出流作为参考
            if ref_flow_id:
                ref_exchange = {
                    "@type": "Exchange",
                    "internalId": len(olca_exchanges) + 1,
                    "flow": {"@type": "Flow", "@id": ref_flow_id},
                    "amount": fu_value,
                    "isInput": False,
                    "isQuantitativeReference": True,
                }
                olca_exchanges.append(ref_exchange)
            else:
                # 如果没有输出流，将第一个流设为参考流
                if olca_exchanges:
                    olca_exchanges[0]["isQuantitativeReference"] = True
            
            process_data = {
                "@type": "Process",
                "@id": process_id,
                "name": f"LCA-LLM Process - {session_id[:8]}",
                "description": f"Auto-generated process from LCA-LLM session. FU: {fu_desc}",
                "processType": "UNIT_PROCESS",
                "exchanges": olca_exchanges,
            }
            
            # 导入到 openLCA
            result = self.olca_client.import_process(process_data)
            
            if result.get("success"):
                return {
                    "success": True,
                    "process_id": process_id,
                    "exchanges_count": len(exchanges)
                }
            else:
                return {
                    "success": False,
                    "error": f"导入 Process 失败: {result.get('error')}"
                }
                
        except Exception as e:
            logger.error(f"创建 openLCA Process 失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _parse_impact_results(self, raw_result: Dict) -> List[Dict]:
        """
        解析 openLCA 计算结果
        
        Args:
            raw_result: openLCA 返回的原始结果
            
        Returns:
            格式化的影响结果列表
        """
        try:
            impact_results = []
            
            # openLCA 返回的结果格式可能因版本而异
            # 常见格式: {"impactResults": [{"impactCategory": {...}, "value": ...}, ...]}
            for ir in raw_result.get("impactResults", []):
                category = ir.get("impactCategory", {})
                impact_results.append({
                    "category": category.get("name", "Unknown"),
                    "value": ir.get("value", 0),
                    "unit": category.get("refUnit", ""),
                    "uuid": category.get("@id", ""),
                })
            
            return impact_results
            
        except Exception as e:
            logger.warning(f"解析影响结果失败: {e}")
            return []
    
    def calculate_lcia(self,
                       session_id: str,
                       lcia_method_uuid: str,
                       flow_mappings: List[Dict] = None) -> Dict[str, Any]:
        """
        执行 LCIA 计算
        
        通过 openLCA IPC 调用计算
        
        Args:
            session_id: 会话 ID
            lcia_method_uuid: LCIA 方法 UUID
            flow_mappings: 流映射
            
        Returns:
            LCIA 计算结果
        """
        self._ensure_initialized()
        
        try:
            # 准备计算参数（启用自动匹配）
            prep = self.prepare_lcia_calculation(
                session_id=session_id,
                lcia_method_uuid=lcia_method_uuid,
                flow_mappings=flow_mappings,
                auto_match=True  # 显式启用自动匹配
            )
            if not prep.get("success"):
                return prep
            
            if not prep.get("ready_for_calculation"):
                return {
                    "success": False,
                    "error": "没有可用于计算的流映射，请先完成 ecoinvent 匹配"
                }
            
            # 检查 openLCA 连接
            olca_test = self.olca_client.test_connection()
            if not olca_test.get("success"):
                return {
                    "success": False,
                    "error": f"openLCA 连接失败: {olca_test.get('error')}"
                }
            
            # 获取 LCI 数据（用于提取 functional_unit）
            lci_data = self.get_session_lci_data(session_id)
            if not lci_data.get("success"):
                return {
                    "success": False,
                    "error": f"获取 LCI 数据失败: {lci_data.get('error')}"
                }
            
            # 完整方案：使用 openLCA IPC 执行实际的 LCIA 计算
            logger.info(f"Starting LCIA calculation with {prep.get('total_mapped')} exchanges")
            
            # Step 1: Create a temporary process in openLCA
            process_result = self._create_process_in_openlca(
                session_id=session_id,
                exchanges=prep.get("exchanges", []),
                functional_unit=lci_data.get("scope", {}).get("functional_unit")
            )
            
            if not process_result.get("success"):
                # Fallback to prepared data if process creation fails
                logger.warning(f"Process creation failed, returning prepared data: {process_result.get('error')}")
                return {
                    "success": True,
                    "session_id": session_id,
                    "lcia_method": prep.get("lcia_method"),
                    "results": {
                        "status": "ready",
                        "message": "LCIA calculation data has been prepared",
                        "exchanges_count": prep.get("total_mapped"),
                        "total_flows": prep.get("total_flows"),
                        "exchanges": prep.get("exchanges", []),
                        "functional_unit": lci_data.get("scope", {}).get("functional_unit"),
                    },
                    "note": f"Process creation failed: {process_result.get('error')}. Returning prepared exchanges data."
                }
            
            process_id = process_result.get("process_id")
            logger.info(f"Process created: {process_id[:30]}...")
            
            # Step 2: Execute LCIA calculation
            calc_result = self.olca_client.calculate(
                target_id=process_id,
                target_type="Process",
                impact_method_id=lcia_method_uuid
            )
            
            if not calc_result.get("success"):
                return {
                    "success": False,
                    "error": f"LCIA calculation failed: {calc_result.get('error')}"
                }
            
            result_id = calc_result.get("result", {}).get("@id")
            logger.info(f"Calculation started, result ID: {result_id[:30]}...")
            
            # Step 3: Wait for calculation to complete
            import time
            max_wait = 30
            for i in range(max_wait):
                state_result = self.olca_client.get_result_state(result_id)
                if state_result.get("success"):
                    state = state_result.get("state", {})
                    if state.get("isReady"):
                        logger.info(f"Calculation complete after {i+1} checks")
                        break
                time.sleep(0.5)
            else:
                return {
                    "success": False,
                    "error": "Calculation timeout after 30 seconds"
                }
            
            # Step 4: Get impact results
            impacts_result = self.olca_client.get_total_impacts(result_id)
            
            if not impacts_result.get("success"):
                return {
                    "success": False,
                    "error": f"Failed to get impact results: {impacts_result.get('error')}"
                }
            
            impacts = impacts_result.get("impacts", [])
            
            # Step 5: Get major flows (simplified version - only show flow amounts)
            flows_result = self.olca_client.get_total_flows(result_id)
            flow_contributions = []
            
            if flows_result.get("success"):
                flows = flows_result.get("flows", [])
                
                # Sort by absolute amount and take top 20
                sorted_flows = sorted(flows, key=lambda x: abs(x.get("amount", 0)), reverse=True)
                
                for i, flow_result in enumerate(sorted_flows[:20]):
                    envi_flow = flow_result.get("enviFlow", {})
                    flow_info = envi_flow.get("flow", {})
                    
                    flow_name = flow_info.get("name", "Unknown")
                    flow_category = flow_info.get("category", "")
                    amount = flow_result.get("amount", 0)
                    unit = flow_info.get("refUnit", "")
                    is_input = envi_flow.get("isInput", False)
                    
                    if abs(amount) > 1e-10:
                        flow_contributions.append({
                            "flow_name": flow_name,
                            "flow_category": flow_category,
                            "flow_amount": amount,
                            "flow_unit": unit,
                            "is_input": is_input
                        })
            
            # Step 6: Format impact results
            formatted_impacts = []
            for impact in impacts:
                impact_cat = impact.get("impactCategory", {})
                formatted_impacts.append({
                    "category": impact_cat.get("name", "Unknown"),
                    "amount": impact.get("amount", 0),
                    "unit": impact_cat.get("refUnit", ""),
                    "description": impact_cat.get("description", "")
                })
            
            # Step 6.5: Get tech flow contributions (for foreground flow analysis)
            tech_flow_contributions = []
            tech_flows_result = self.olca_client.get_tech_flow_contributions(result_id)
            
            if tech_flows_result.get("success"):
                tech_flows = tech_flows_result.get("tech_flows", [])
                logger.info(f"Found {len(tech_flows)} tech flows")
                
                # Extract tech flow information
                for tf in tech_flows[:50]:  # Limit to top 50
                    tech_flow_info = tf.get("techFlow", {})
                    provider = tech_flow_info.get("provider", {})
                    flow = tech_flow_info.get("flow", {})
                    
                    tech_flow_contributions.append({
                        "flow_name": flow.get("name", "Unknown"),
                        "provider_name": provider.get("name", "Unknown"),
                        "amount": tf.get("amount", 0)
                    })
            
            # Step 7: Dispose result to free memory
            self.olca_client.dispose_result(result_id)
            
            return {
                "success": True,
                "session_id": session_id,
                "lcia_method": prep.get("lcia_method"),
                "results": {
                    "status": "completed",
                    "message": "LCIA calculation completed successfully",
                    "exchanges_count": prep.get("total_mapped"),
                    "total_flows": prep.get("total_flows"),
                    "functional_unit": lci_data.get("scope", {}).get("functional_unit"),
                    "impacts": formatted_impacts,
                    "impact_count": len(formatted_impacts),
                    "flow_contributions": flow_contributions,
                    "flow_contributions_count": len(flow_contributions),
                    "tech_flow_contributions": tech_flow_contributions,
                    "tech_flow_contributions_count": len(tech_flow_contributions)
                }
            }
            
        except Exception as e:
            logger.error(f"LCIA 计算失败: {e}")
            return {"success": False, "error": str(e)}


# 全局实例
_calculator_instance = None

def get_lcia_calculator() -> LCIACalculator:
    """获取全局计算器实例"""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = LCIACalculator()
    return _calculator_instance
