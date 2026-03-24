"""
可搜索单位选择器组件
Searchable Unit Selector Component

功能:
1. 可搜索的下拉菜单
2. 标准单位选择
3. 自定义单位输入
4. 动态显示/隐藏文本框
"""

import streamlit as st
import sys
import os
from typing import Optional, Tuple

# 添加后端路径
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend')
sys.path.append(backend_path)

try:
    from utils.unit_processor import get_unit_dropdown_options, search_units, validate_unit
except ImportError:
    # 如果导入失败，提供备用选项
    def get_unit_dropdown_options():
        return [
            {'value': 'kg', 'label': 'kg (kilogram)', 'category': 'Mass'},
            {'value': 'kWh', 'label': 'kWh (kilowatt hour)', 'category': 'Energy'},
            {'value': 'L', 'label': 'L (liter)', 'category': 'Volume'},
            {'value': '__custom__', 'label': '-- Other (Custom Unit) --', 'category': 'Custom'}
        ]
    
    def search_units(query: str, limit: int = 10):
        return get_unit_dropdown_options()
    
    def validate_unit(unit: str):
        return True, "Valid unit"

class UnitSelector:
    """单位选择器类"""
    
    def __init__(self, key_prefix: str = "unit_selector"):
        self.key_prefix = key_prefix
        self.options = get_unit_dropdown_options()
    
    def render(self, 
               label: str = "Unit", 
               default_value: Optional[str] = None,
               help_text: Optional[str] = None,
               required: bool = True) -> Tuple[str, bool]:
        """
        渲染单位选择器
        
        Args:
            label: 标签文本
            default_value: 默认值
            help_text: 帮助文本
            required: 是否必填
            
        Returns:
            Tuple[str, bool]: (选中的单位, 是否有效)
        """
        
        # 创建选项列表 (value -> label 映射)
        option_labels = [opt['label'] for opt in self.options]
        option_values = [opt['value'] for opt in self.options]
        
        # 查找默认选项索引
        default_index = 0
        if default_value:
            try:
                default_index = option_values.index(default_value)
            except ValueError:
                # 如果默认值不在标准选项中，设为自定义
                default_index = len(option_values) - 1  # 最后一个是 "Custom"
        
        # 主选择框
        selected_label = st.selectbox(
            label=label,
            options=option_labels,
            index=default_index,
            help=help_text,
            key=f"{self.key_prefix}_selectbox"
        )
        
        # 获取选中的值
        selected_index = option_labels.index(selected_label)
        selected_value = option_values[selected_index]
        
        # 如果选择了自定义选项，显示文本输入框
        final_unit = selected_value
        is_valid = True
        
        if selected_value == '__custom__':
            # 自定义单位输入框
            custom_unit = st.text_input(
                label="Custom Unit",
                value=default_value if default_value and default_value not in option_values else "",
                placeholder="Enter custom unit (e.g., kg/m², kg CO2-eq)",
                key=f"{self.key_prefix}_custom_input",
                help="Enter a custom unit not available in the dropdown list"
            )
            
            if custom_unit.strip():
                final_unit = custom_unit.strip()
                is_valid, validation_msg = validate_unit(final_unit)
                
                if not is_valid and required:
                    st.error(f"Invalid unit: {validation_msg}")
                elif not is_valid:
                    st.warning(f"Warning: {validation_msg}")
            else:
                if required:
                    st.error("Please enter a custom unit")
                    is_valid = False
                final_unit = ""
        
        else:
            # 标准单位选择 - 不显示类别信息
            pass
        
        return final_unit, is_valid
    
    def render_with_search(self, 
                          label: str = "Unit",
                          default_value: Optional[str] = None,
                          help_text: Optional[str] = None,
                          required: bool = True) -> Tuple[str, bool]:
        """
        渲染带搜索功能的单位选择器
        
        Args:
            label: 标签文本
            default_value: 默认值
            help_text: 帮助文本
            required: 是否必填
            
        Returns:
            Tuple[str, bool]: (选中的单位, 是否有效)
        """
        
        # 搜索框
        search_query = st.text_input(
            label=f"Search {label}",
            placeholder="Type to search units (e.g., 'energy', 'kWh', 'mass')",
            key=f"{self.key_prefix}_search"
        )
        
        # 根据搜索结果过滤选项
        if search_query:
            filtered_options = search_units(search_query, limit=20)
        else:
            filtered_options = self.options
        
        # 创建过滤后的选项列表
        option_labels = [opt['label'] for opt in filtered_options]
        option_values = [opt['value'] for opt in filtered_options]
        
        # 查找默认选项索引
        default_index = 0
        if default_value and default_value in option_values:
            default_index = option_values.index(default_value)
        
        # 选择框
        if option_labels:
            selected_label = st.selectbox(
                label=f"Select {label}",
                options=option_labels,
                index=default_index,
                help=help_text,
                key=f"{self.key_prefix}_filtered_selectbox"
            )
            
            selected_index = option_labels.index(selected_label)
            selected_value = option_values[selected_index]
        else:
            st.warning("No units found matching your search")
            selected_value = '__custom__'
        
        # 处理选择结果
        final_unit = selected_value
        is_valid = True
        
        if selected_value == '__custom__':
            # 自定义单位输入
            custom_unit = st.text_input(
                label="Custom Unit",
                value=default_value if default_value and default_value not in option_values else search_query,
                placeholder="Enter custom unit (e.g., kg CO2-eq)",
                key=f"{self.key_prefix}_search_custom_input"
            )
            
            if custom_unit.strip():
                final_unit = custom_unit.strip()
                is_valid, validation_msg = validate_unit(final_unit)
                
                if not is_valid and required:
                    st.error(f"Invalid unit: {validation_msg}")
                elif not is_valid:
                    st.warning(f"Warning: {validation_msg}")
            else:
                if required:
                    st.error("Please enter a custom unit")
                    is_valid = False
                final_unit = ""
        else:
            # 标准单位选择 - 不显示类别信息
            pass
        
        return final_unit, is_valid

def render_unit_selector(label: str = "Unit", 
                        default_value: Optional[str] = None,
                        key_prefix: str = "unit",
                        search_enabled: bool = True,
                        required: bool = True) -> Tuple[str, bool]:
    """
    便捷函数：渲染单位选择器
    
    Args:
        label: 标签文本
        default_value: 默认值
        key_prefix: 组件key前缀
        search_enabled: 是否启用搜索功能
        required: 是否必填
        
    Returns:
        Tuple[str, bool]: (选中的单位, 是否有效)
    """
    # 简化版本，直接在这里实现，避免复杂的类结构
    try:
        from utils.unit_processor import get_unit_dropdown_options, validate_unit
    except ImportError:
        # 备用选项
        def get_unit_dropdown_options():
            return [
                {'value': 'kg', 'label': 'kg (kilogram)', 'category': 'Mass'},
                {'value': 'kWh', 'label': 'kWh (kilowatt hour)', 'category': 'Energy'},
                {'value': '__custom__', 'label': '-- Other (Custom Unit) --', 'category': 'Custom'}
            ]
        def validate_unit(unit: str):
            return True, "Valid unit"
    
    # 获取选项
    options = get_unit_dropdown_options()
    option_labels = [opt['label'] for opt in options]
    option_values = [opt['value'] for opt in options]
    
    # 查找默认选项索引
    default_index = 0
    if default_value:
        try:
            default_index = option_values.index(default_value)
        except ValueError:
            # 如果默认值不在标准选项中，设为自定义
            default_index = len(option_values) - 1  # 最后一个是 "Custom"
    
    # 主选择框
    selected_label = st.selectbox(
        label=label,
        options=option_labels,
        index=default_index,
        key=f"{key_prefix}_selectbox"
    )
    
    # 获取选中的值
    selected_index = option_labels.index(selected_label)
    selected_value = option_values[selected_index]
    
    # 处理选择结果
    final_unit = selected_value
    is_valid = True
    
    if selected_value == '__custom__':
        # 自定义单位输入框
        custom_unit = st.text_input(
            label="Custom Unit",
            value=default_value if default_value and default_value not in option_values else "",
            placeholder="Enter custom unit (e.g., kg/m², kg CO₂-eq)",
            key=f"{key_prefix}_custom_input",
            help="Enter a custom unit not available in the dropdown list"
        )
        
        if custom_unit.strip():
            final_unit = custom_unit.strip()
            is_valid, validation_msg = validate_unit(final_unit)
            
            if not is_valid and required:
                st.error(f"Invalid unit: {validation_msg}")
            elif not is_valid:
                st.warning(f"Warning: {validation_msg}")
        else:
            if required:
                st.error("Please enter a custom unit")
                is_valid = False
            final_unit = ""
    
    return final_unit, is_valid

# 示例用法
if __name__ == "__main__":
    st.title("Unit Selector Demo")
    
    st.subheader("Basic Unit Selector")
    unit1, valid1 = render_unit_selector(
        label="Select Unit",
        key_prefix="demo1",
        search_enabled=False
    )
    st.write(f"Selected: {unit1}, Valid: {valid1}")
    
    st.subheader("Searchable Unit Selector")
    unit2, valid2 = render_unit_selector(
        label="Search and Select Unit",
        key_prefix="demo2",
        search_enabled=True
    )
    st.write(f"Selected: {unit2}, Valid: {valid2}")
