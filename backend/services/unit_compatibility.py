"""
单位兼容性检查模块
用于在流量匹配时检查单位是否兼容
"""
from typing import Dict, List, Optional

# 定义单位兼容性组
UNIT_GROUPS = {
    'mass': ['kg', 'g', 'mg', 't', 'ton', 'tonne', 'lb', 'oz'],
    'energy': ['mj', 'kwh', 'kj', 'wh', 'j', 'cal', 'kcal', 'btu'],
    'volume': ['m3', 'l', 'ml', 'dm3', 'cm3', 'gal', 'ft3'],
    'area': ['m2', 'km2', 'cm2', 'mm2', 'ha', 'ft2', 'acre'],
    'length': ['m', 'km', 'cm', 'mm', 'ft', 'in', 'mile'],
    'radioactivity': ['bq', 'kbq', 'mbq', 'gbq', 'ci', 'mci'],
    'time': ['s', 'min', 'h', 'hr', 'hour', 'd', 'day', 'year', 'a'],
    'count': ['item', 'items', 'item(s)', 'unit', 'units', 'p', 'piece', 'pieces'],
}

# 单位转换因子（转换到基础单位）
CONVERSION_FACTORS = {
    # 质量 -> kg
    'kg': 1.0,
    'g': 0.001,
    'mg': 0.000001,
    't': 1000.0,
    'ton': 1000.0,
    'tonne': 1000.0,
    
    # 能量 -> MJ
    'mj': 1.0,
    'kwh': 3.6,  # 1 kWh = 3.6 MJ
    'kj': 0.001,
    'wh': 0.0036,
    'j': 0.000001,
    
    # 体积 -> m3
    'm3': 1.0,
    'l': 0.001,
    'ml': 0.000001,
    'dm3': 0.001,
    'cm3': 0.000001,
    
    # 面积 -> m2
    'm2': 1.0,
    'km2': 1000000.0,
    'cm2': 0.0001,
    'mm2': 0.000001,
    'ha': 10000.0,
}


def normalize_unit(unit: str) -> str:
    """
    标准化单位字符串
    
    Args:
        unit: 原始单位字符串
        
    Returns:
        标准化后的单位（小写，去除空格）
    """
    if not unit:
        return ''
    return unit.lower().strip().replace(' ', '')


def get_unit_group(unit: str) -> Optional[str]:
    """
    获取单位所属的组
    
    Args:
        unit: 单位字符串
        
    Returns:
        单位组名称，如果未找到则返回 None
    """
    normalized = normalize_unit(unit)
    
    for group_name, units in UNIT_GROUPS.items():
        if normalized in units:
            return group_name
    
    return None


def are_units_compatible(unit1: str, unit2: str) -> bool:
    """
    检查两个单位是否兼容（属于同一组）
    
    Args:
        unit1: 第一个单位
        unit2: 第二个单位
        
    Returns:
        True 如果兼容，False 否则
    """
    if not unit1 or not unit2:
        return False
    
    group1 = get_unit_group(unit1)
    group2 = get_unit_group(unit2)
    
    if not group1 or not group2:
        # 如果单位不在已知组中，检查是否完全相同
        return normalize_unit(unit1) == normalize_unit(unit2)
    
    return group1 == group2


def convert_unit(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """
    转换单位
    
    Args:
        value: 原始数值
        from_unit: 源单位
        to_unit: 目标单位
        
    Returns:
        转换后的数值，如果无法转换则返回 None
    """
    if not are_units_compatible(from_unit, to_unit):
        return None
    
    from_normalized = normalize_unit(from_unit)
    to_normalized = normalize_unit(to_unit)
    
    if from_normalized == to_normalized:
        return value
    
    # 获取转换因子
    from_factor = CONVERSION_FACTORS.get(from_normalized)
    to_factor = CONVERSION_FACTORS.get(to_normalized)
    
    if from_factor is None or to_factor is None:
        # 无法转换，但单位兼容（可能是不常见的单位）
        return None
    
    # 转换：value * from_factor / to_factor
    return value * from_factor / to_factor


def get_compatible_units(unit: str) -> List[str]:
    """
    获取与给定单位兼容的所有单位
    
    Args:
        unit: 单位字符串
        
    Returns:
        兼容单位列表
    """
    group = get_unit_group(unit)
    if not group:
        return [normalize_unit(unit)]
    
    return UNIT_GROUPS.get(group, [])


def suggest_unit_conversion(original_value: float, original_unit: str, 
                           matched_unit: str) -> Dict:
    """
    建议单位转换
    
    Args:
        original_value: 原始数值
        original_unit: 原始单位
        matched_unit: 匹配到的单位
        
    Returns:
        包含转换信息的字典
    """
    result = {
        'compatible': are_units_compatible(original_unit, matched_unit),
        'original_value': original_value,
        'original_unit': original_unit,
        'matched_unit': matched_unit,
        'converted_value': None,
        'conversion_factor': None,
        'suggestion': ''
    }
    
    if not result['compatible']:
        result['suggestion'] = f"单位不兼容：{original_unit} 和 {matched_unit} 不属于同一类型"
        return result
    
    # 尝试转换
    converted = convert_unit(original_value, original_unit, matched_unit)
    
    if converted is not None:
        result['converted_value'] = converted
        result['conversion_factor'] = converted / original_value if original_value != 0 else None
        result['suggestion'] = f"可以转换：{original_value} {original_unit} = {converted:.4f} {matched_unit}"
    else:
        result['suggestion'] = f"单位兼容但无法自动转换，需要手动处理"
    
    return result


# 示例使用
if __name__ == "__main__":
    # 测试单位兼容性
    print("测试单位兼容性:")
    print(f"kg 和 g: {are_units_compatible('kg', 'g')}")  # True
    print(f"kWh 和 MJ: {are_units_compatible('kWh', 'MJ')}")  # True
    print(f"kg 和 m2: {are_units_compatible('kg', 'm2')}")  # False
    print(f"kg 和 kBq: {are_units_compatible('kg', 'kBq')}")  # False
    
    print("\n测试单位转换:")
    print(f"67.47 kWh -> MJ: {convert_unit(67.47, 'kWh', 'MJ')}")  # 242.892
    print(f"4.11 kg -> g: {convert_unit(4.11, 'kg', 'g')}")  # 4110
    
    print("\n测试转换建议:")
    suggestion = suggest_unit_conversion(67.47, 'kWh', 'MJ')
    print(suggestion['suggestion'])
