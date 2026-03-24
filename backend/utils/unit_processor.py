"""
单位处理工具类
Unit Processing Utilities

功能:
1. 加载标准单位库 (用于UI下拉菜单)
2. 加载完整单位库 (用于后端识别)
3. 提供单位标准化和验证功能
4. 支持智能单位匹配
"""

import yaml
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class UnitProcessor:
    """单位处理器"""
    
    def __init__(self):
        self.resources_path = Path(__file__).parent.parent.parent / "resources"
        self.standard_units = self._load_standard_units()
        # 暂时不加载完整单位库和映射，专家工作台阶段不需要
        # self.full_units = self._load_full_units()
        # self.unit_mapping = self._create_unit_mapping()
    
    def _load_standard_units(self) -> Dict:
        """加载标准单位库 (用于UI显示)"""
        try:
            with open(self.resources_path / "standard_units.yml", 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading standard units: {e}")
            return {}
    
    def _load_full_units(self) -> Dict:
        """加载完整单位库 (用于后端识别)"""
        try:
            with open(self.resources_path / "unit.yml", 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading full units: {e}")
            return {}
    
    def _create_unit_mapping(self) -> Dict[str, str]:
        """创建从完整单位到标准单位的映射"""
        mapping = {}
        
        # 为每个标准单位创建映射
        for category_name, units in self.standard_units.items():
            if isinstance(units, list):
                for unit_info in units:
                    if isinstance(unit_info, dict) and 'unit' in unit_info:
                        standard_unit = unit_info['unit']
                        # 标准单位映射到自己
                        mapping[standard_unit] = standard_unit
                        
                        # 查找对应的完整单位列表
                        full_category = self._find_full_category(category_name, standard_unit)
                        if full_category:
                            for variant in full_category:
                                mapping[variant.lower()] = standard_unit
        
        return mapping
    
    def _find_full_category(self, category_name: str, standard_unit: str) -> Optional[List[str]]:
        """在完整单位库中查找对应的单位变体"""
        # 根据类别名称和标准单位，在full_units中查找对应的变体列表
        category_mapping = {
            'mass_units': ['mass_units'],
            'volume_units': ['volume_units'],
            'energy_units': ['energy_units'],
            'power_units': ['power_units'],
            'length_units': ['length_units'],
            'area_units': ['area_units'],
            'time_units': ['time_units'],
            'temperature_units': ['temperature_units'],
            'pressure_units': ['pressure_units'],
            'flow_rate_units': ['flow_rate_units'],
            'concentration_units': ['concentration_units'],
            'cost_units': ['cost_units'],
            'dimensionless_units': ['dimensionless_units']
        }
        
        if category_name in category_mapping:
            for full_category_name in category_mapping[category_name]:
                if full_category_name in self.full_units:
                    category_data = self.full_units[full_category_name]
                    
                    # 处理嵌套结构 (如 electrical, thermal 等子类别)
                    if isinstance(category_data, dict):
                        for subcategory, units in category_data.items():
                            if isinstance(units, list) and self._contains_unit(units, standard_unit):
                                return units
                    elif isinstance(category_data, list) and self._contains_unit(category_data, standard_unit):
                        return category_data
        
        return None
    
    def _contains_unit(self, unit_list: List[str], target_unit: str) -> bool:
        """检查单位列表是否包含目标单位"""
        return any(unit.lower() == target_unit.lower() for unit in unit_list)
    
    def get_dropdown_options(self) -> List[Dict[str, str]]:
        """获取下拉菜单选项"""
        options = []
        
        for category_name, units in self.standard_units.items():
            if isinstance(units, list):
                for unit_info in units:
                    if isinstance(unit_info, dict):
                        options.append({
                            'value': unit_info['unit'],
                            'label': unit_info['display_name'],
                            'category': unit_info.get('category', 'Other')
                        })
        
        # 按类别和单位名称排序
        options.sort(key=lambda x: (x['category'], x['label']))
        
        # 添加自定义选项
        options.append({
            'value': '__custom__',
            'label': '-- Other (Custom Unit) --',
            'category': 'Custom'
        })
        
        return options
    
    def get_grouped_options(self) -> Dict[str, List[Dict[str, str]]]:
        """获取按类别分组的下拉菜单选项"""
        options = self.get_dropdown_options()
        grouped = {}
        
        for option in options:
            category = option['category']
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(option)
        
        return grouped
    
    def normalize_unit(self, input_unit: str) -> str:
        """将输入单位标准化为标准单位 - 简化版本，直接返回输入"""
        # 专家工作台阶段：直接返回用户输入的单位
        # 后续LLM阶段可能需要更复杂的映射逻辑
        return input_unit.strip() if input_unit else ""
    
    def validate_unit(self, unit: str) -> Tuple[bool, str]:
        """验证单位是否有效 - 简化版本"""
        if not unit:
            return False, "Unit cannot be empty"
        
        # 检查是否为标准单位
        standard_units = [opt['value'] for opt in self.get_dropdown_options() if opt['value'] != '__custom__']
        if unit in standard_units:
            return True, "Valid standard unit"
        
        # 自定义单位 - 专家工作台阶段接受任何非空单位
        if len(unit.strip()) > 0:
            return True, "Custom unit"
        
        return False, "Invalid unit"
    
    def search_units(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """搜索单位 (用于可搜索下拉菜单)"""
        if not query:
            return self.get_dropdown_options()[:limit]
        
        query_lower = query.lower()
        options = self.get_dropdown_options()
        
        # 搜索匹配
        matches = []
        for option in options:
            if (query_lower in option['label'].lower() or 
                query_lower in option['value'].lower() or
                query_lower in option['category'].lower()):
                matches.append(option)
        
        return matches[:limit]

# 全局实例
unit_processor = UnitProcessor()

def get_unit_dropdown_options():
    """获取单位下拉菜单选项 (供前端调用)"""
    return unit_processor.get_dropdown_options()

def get_grouped_unit_options():
    """获取分组的单位选项"""
    return unit_processor.get_grouped_options()

def normalize_unit(unit: str) -> str:
    """标准化单位"""
    return unit_processor.normalize_unit(unit)

def validate_unit(unit: str) -> Tuple[bool, str]:
    """验证单位"""
    return unit_processor.validate_unit(unit)

def search_units(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """搜索单位"""
    return unit_processor.search_units(query, limit)
