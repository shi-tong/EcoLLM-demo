#!/usr/bin/env python3
"""
决策特征计算器
Decision Features Calculator

根据Expert_Workbench Decision Logic Schema v1.3和Q&A.md规范实现
自动计算候选文本块的决策特征，用于专家决策支持

功能:
1. unit_hits: 数字附近的单位匹配次数
2. pattern_count: 匹配到的"值+单位"对数量  
3. quantitative_pattern_score: "值+单位"占chunk长度比
4. matched_examples: 抽样示例（前2-3个命中短语）
5. contains_basis_tokens: 是否出现基准词
6. is_table: 是否为表格（从分块器获取或启发式判断）
"""

import re
import yaml
from typing import Dict, List, Any, Tuple, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DecisionFeaturesCalculator:
    """决策特征计算器"""
    
    def __init__(self):
        """初始化计算器，加载单位库"""
        self.resources_path = Path(__file__).parent.parent.parent / "resources"
        self.units_data = self._load_units()
        self.unit_pattern = self._create_unit_pattern()
        self.value_pattern = self._create_value_pattern()
        self.combined_pattern = self._create_combined_pattern()
        self.basis_pattern = self._create_basis_pattern()
        
    def _load_units(self) -> Dict:
        """加载增材制造专用单位库"""
        try:
            # 优先使用增材制造专用单位库
            with open(self.resources_path / "unit_am.yml", 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load unit_am.yml: {e}")
            try:
                # 备用：使用通用单位库
                with open(self.resources_path / "unit.yml", 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e2:
                logger.warning(f"Failed to load unit.yml: {e2}")
                # 最后备用：硬编码增材制造核心单位
                return {
                    'mass_units': {
                        'metric': ['kg', 'g', 't', 'mg', 'kilogram', 'gram', 'tonne']
                    },
                    'volume_units': {
                        'metric': ['L', 'l', 'mL', 'ml', 'm³', 'm3', 'cm³', 'cm3', 'mm³', 'mm3']
                    },
                    'energy_units': {
                        'electrical': ['kWh', 'Wh', 'MWh', 'kilowatt hour'],
                        'thermal': ['J', 'kJ', 'MJ', 'GJ', 'joule', 'kilojoule', 'megajoule']
                    },
                    'power_units': ['W', 'kW', 'MW', 'watt', 'kilowatt'],
                    'length_units': {
                        'metric': ['m', 'cm', 'mm', 'μm', 'meter', 'millimeter']
                    },
                    'temperature_units': ['°C', '℃', 'K', 'Celsius', 'Kelvin']
                }
    
    def _create_unit_pattern(self) -> str:
        """创建单位正则表达式"""
        all_units = []
        
        # 从unit.yml提取所有单位
        for category, subcategories in self.units_data.items():
            if isinstance(subcategories, dict):
                for subcat, units in subcategories.items():
                    if isinstance(units, list):
                        all_units.extend(units)
            elif isinstance(subcategories, list):
                all_units.extend(subcategories)
        
        # 过滤掉过短或可能误匹配的单位
        filtered_units = []
        exclude_patterns = ['per', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'with']  # 排除常见介词
        
        for unit in all_units:
            # 排除过短的单位（除了一些标准缩写）
            if len(unit) == 1 and unit.lower() not in ['l', 't', 'g', 'h', 'm', 'w', 'j', 'v', 'a']:
                continue
            # 排除常见介词
            if unit.lower() in exclude_patterns:
                continue
            # 排除纯数字
            if unit.isdigit():
                continue
            filtered_units.append(unit)
        
        # 去重并按长度排序（长的在前，避免短单位匹配问题）
        unique_units = list(set(filtered_units))
        unique_units.sort(key=len, reverse=True)
        
        # 转义特殊字符并构建正则
        escaped_units = [re.escape(unit) for unit in unique_units]
        
        return r"(?:" + "|".join(escaped_units) + r")"
    
    def _create_value_pattern(self) -> str:
        """创建数值正则表达式"""
        # 支持千分位逗号/空格，小数点，科学记数法
        return r"(?<!\d)(\d{1,3}(?:[ ,]\d{3})*(?:\.\d+)?|\d+\.\d+|\d+(?:[eE][+-]?\d+)?)(?!\d)"
    
    def _create_combined_pattern(self) -> re.Pattern:
        """创建值+单位组合正则表达式"""
        pattern = self.value_pattern + r"\s*" + self.unit_pattern + r"\b"
        return re.compile(pattern, flags=re.IGNORECASE)
    
    def _create_basis_pattern(self) -> re.Pattern:
        """创建基准词正则表达式"""
        # 基于Q&A.md中的BASIS正则
        basis_pattern = r"(?:\bper\b\s+(?:part|batch|build|unit|hour|day|tonne|kg)|/t\b|/kg\b|/h\b|each\b|per\s+batch|per\s+hour)"
        return re.compile(basis_pattern, flags=re.IGNORECASE)
    
    def calculate_features(self, text: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        计算文本块的决策特征（改进版 - 支持表格分离式计算）
        
        Args:
            text: 文本内容
            metadata: 元数据（可能包含is_table信息）
            
        Returns:
            Dict: 决策特征字典
        """
        if not text or not isinstance(text, str):
            return self._empty_features()
        
        # 预处理文本
        cleaned_text = self._preprocess_text(text)
        
        # 首先检测是否为表格
        is_table = self._detect_table(text, metadata)
        
        # 根据是否为表格选择不同的计算策略
        if is_table:
            # 表格模式：分离式计算
            pattern_count, matched_examples = self._calculate_table_patterns(cleaned_text)
        else:
            # 叙述模式：组合式计算（原逻辑）
            matches = list(self.combined_pattern.finditer(cleaned_text))
            pattern_count = len(matches)
            matched_examples = self._extract_examples(matches)
        
        # 计算改进的quantitative_pattern_score（基于词密度而非字符密度）
        quantitative_pattern_score = self._calculate_qps_v2(pattern_count, cleaned_text)
        
        # 检测基准词
        contains_basis_tokens = bool(self.basis_pattern.search(cleaned_text))
        
        return {
            # 移除unit_hits，只保留pattern_count
            "pattern_count": pattern_count,
            "quantitative_pattern_score": round(quantitative_pattern_score, 3),
            "matched_examples": matched_examples,
            "contains_basis_tokens": contains_basis_tokens,
            "is_table": is_table
        }
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 规范化空白字符，保留千分位逗号/空格
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _calculate_unit_hits(self, matches: List[re.Match]) -> int:
        """计算单位命中数（不去重，每次匹配都计数）"""
        unit_count = 0
        for match in matches:
            # 提取单位部分（匹配组的后半部分）
            full_match = match.group(0)
            # 使用单位模式重新匹配以提取单位
            unit_match = re.search(self.unit_pattern, full_match, re.IGNORECASE)
            if unit_match:
                unit_count += 1
        return unit_count
    
    def _calculate_qps(self, matches: List[re.Match], text: str) -> float:
        """计算量化模式分数"""
        if not text:
            return 0.0
        
        total_match_length = sum(len(match.group(0)) for match in matches)
        text_length = len(text)
        
        if text_length == 0:
            return 0.0
        
        qps = total_match_length / text_length
        return min(qps, 1.0)  # 限制在0-1范围内
    
    def _extract_examples(self, matches: List[re.Match], limit: int = 3) -> List[str]:
        """提取匹配示例"""
        examples = []
        seen = set()
        
        for match in matches:
            example = match.group(0).strip()
            # 截断过长的示例
            if len(example) > 20:
                example = example[:17] + "..."
            
            # 去重（忽略大小写和空格）
            normalized = re.sub(r'\s+', ' ', example.lower())
            if normalized not in seen:
                examples.append(example)
                seen.add(normalized)
                
                if len(examples) >= limit:
                    break
        
        return examples
    
    def _calculate_table_patterns(self, text: str) -> Tuple[int, List[str]]:
        """
        表格模式：分离式计算pattern_count（改进版，修复复合单位和标题数值问题）
        
        Args:
            text: 预处理后的文本
            
        Returns:
            Tuple[int, List[str]]: (pattern_count, matched_examples)
        """
        # 过滤数值匹配，排除标题中的数字
        raw_number_matches = list(re.finditer(self.value_pattern, text))
        filtered_number_matches = self._filter_table_numbers(raw_number_matches, text)
        
        # 识别复合单位（如L/min作为整体）
        compound_units = self._identify_compound_units(text)
        
        # 如果有复合单位，使用复合单位逻辑
        if compound_units:
            pattern_count = min(len(filtered_number_matches), len(compound_units))
            examples = self._build_compound_unit_examples(filtered_number_matches, compound_units, pattern_count)
        else:
            # 回退到分离单位逻辑
            raw_unit_matches = list(re.finditer(self.unit_pattern, text, re.IGNORECASE))
            filtered_unit_matches = self._filter_table_units(raw_unit_matches, filtered_number_matches, text)
            pattern_count = min(len(filtered_number_matches), len(filtered_unit_matches))
            examples = self._build_table_examples(filtered_number_matches, filtered_unit_matches, pattern_count)
        
        return pattern_count, examples
    
    def _filter_table_units(self, unit_matches: List[re.Match], number_matches: List[re.Match], text: str) -> List[re.Match]:
        """
        过滤表格中的单位匹配，防止误检
        
        Args:
            unit_matches: 原始单位匹配列表
            number_matches: 数值匹配列表
            text: 文本内容
            
        Returns:
            List[re.Match]: 过滤后的单位匹配列表
        """
        if not unit_matches:
            return []
            
        filtered_matches = []
        number_positions = [(m.start(), m.end()) for m in number_matches]
        
        # 统计单位出现频率，过滤高频噪音
        unit_frequency = {}
        for match in unit_matches:
            unit_text = match.group(0).lower()
            unit_frequency[unit_text] = unit_frequency.get(unit_text, 0) + 1
        
        for unit_match in unit_matches:
            unit_text = unit_match.group(0)
            unit_start = unit_match.start()
            unit_end = unit_match.end()
            
            # 规则1: 多字符单位优先保留（如 kg, cm³, L/min, min）
            if len(unit_text) >= 2:
                # 但要排除明显的噪音（如出现在标题中的词汇）
                if unit_frequency.get(unit_text.lower(), 0) <= 3:  # 出现次数不超过3次
                    filtered_matches.append(unit_match)
                continue
            
            # 规则2: 单字符单位需要严格验证
            if len(unit_text) == 1:
                # 2a. 必须是核心单位缩写
                if unit_text.lower() not in ['l', 'g', 't', 'm']:  # 只保留最核心的单位
                    continue
                
                # 2b. 上下文验证：检查是否在复合单位中（如L/min, g/cm³）
                context_start = max(0, unit_start - 2)
                context_end = min(len(text), unit_end + 5)
                context = text[context_start:context_end]
                is_in_compound_unit = '/' in context or '²' in context or '³' in context
                
                # 2c. 位置验证：必须紧邻数值（±5字符内）
                is_very_close_to_number = any(
                    abs(unit_start - num_end) <= 5 or abs(unit_end - num_start) <= 5
                    for num_start, num_end in number_positions
                )
                
                # 2d. 或者在明确的单位列中（检查周围是否有"Unit"标题）
                context_start = max(0, unit_start - 30)
                context_end = min(len(text), unit_end + 30)
                context = text[context_start:context_end].lower()
                is_in_unit_column = 'unit' in context and '|' in context
                
                # 2e. 频率过滤：复合单位中的核心字母可以豁免频率限制
                if is_in_compound_unit:
                    # 复合单位中的核心单位字母（L, g, m等）豁免频率限制
                    pass  # 不进行频率过滤
                else:
                    # 非复合单位的单字符需要严格频率限制
                    if unit_frequency.get(unit_text.lower(), 0) > 2:
                        continue
                
                if is_very_close_to_number or is_in_unit_column or is_in_compound_unit:
                    filtered_matches.append(unit_match)
        
        return filtered_matches
    
    def _build_table_examples(self, number_matches: List[re.Match], unit_matches: List[re.Match], pattern_count: int) -> List[str]:
        """
        构建表格模式的matched_examples
        
        Args:
            number_matches: 数值匹配列表
            unit_matches: 过滤后的单位匹配列表
            pattern_count: 模式计数
            
        Returns:
            List[str]: 示例列表
        """
        examples = []
        
        # 如果有有效的数值-单位配对，优先显示配对信息
        if pattern_count > 0:
            # 显示前几个数值（标注为数值）
            for i, match in enumerate(number_matches[:min(3, pattern_count)]):
                examples.append(f"{match.group(0)}")
            
            # 如果有单位，显示单位信息
            if unit_matches:
                unit_texts = [m.group(0) for m in unit_matches[:3]]
                unique_units = list(dict.fromkeys(unit_texts))  # 去重保序
                if len(unique_units) == 1:
                    examples.append(f"({unique_units[0]})")
                elif len(unique_units) <= 3:
                    examples.append(f"({', '.join(unique_units)})")
        else:
            # 如果没有有效配对，显示原始数值
            for match in number_matches[:3]:
                examples.append(match.group(0))
        
        return examples[:3]  # 最多返回3个示例
    
    def _filter_table_numbers(self, number_matches: List[re.Match], text: str) -> List[re.Match]:
        """
        过滤表格中的数值匹配，排除标题、编号等非数据数值
        
        Args:
            number_matches: 原始数值匹配列表
            text: 文本内容
            
        Returns:
            List[re.Match]: 过滤后的数值匹配列表
        """
        filtered_matches = []
        
        for number_match in number_matches:
            number_start = number_match.start()
            number_end = number_match.end()
            
            # 检查上下文，排除标题中的数字
            context_start = max(0, number_start - 15)
            context_end = min(len(text), number_end + 15)
            context = text[context_start:context_end]
            
            # 排除条件：
            # 1. 在标题中（## Table, # Figure等）
            if re.search(r'##?\s*[Tt]able|##?\s*[Ff]igure', context):
                continue
            
            # 2. 在章节编号中（1.1, 2.3.4等模式，且前后没有表格分隔符）
            if re.search(r'\b\d+\.\d+(\.\d+)?\b', number_match.group()) and '|' not in context:
                continue
            
            # 3. 在页码或引用中
            if re.search(r'[Pp]age|[Pp]\.|ref|cite', context, re.IGNORECASE):
                continue
            
            # 保留在表格数据行中的数值（有|分隔符的上下文）
            filtered_matches.append(number_match)
        
        return filtered_matches
    
    def _identify_compound_units(self, text: str) -> List[Dict[str, Any]]:
        """
        识别复合单位（如L/min, g/cm³, kWh/kg等）
        
        Args:
            text: 文本内容
            
        Returns:
            List[Dict]: 复合单位信息列表，包含匹配对象和单位文本
        """
        compound_units = []
        
        # 复合单位模式：单位/单位, 单位²，单位³等
        compound_patterns = [
            r'\b[A-Za-z]+/[A-Za-z]+\b',  # L/min, kg/m³, kWh/kg
            r'\b[A-Za-z]+[²³]\b',        # cm², m³, km²
            r'\b[A-Za-z]+\s*/\s*[A-Za-z]+\b',  # L / min (带空格)
        ]
        
        seen_units = set()  # 去重
        
        for pattern in compound_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                unit_text = match.group(0)
                # 验证是否为真实的单位组合且未重复
                if self._is_valid_compound_unit(unit_text) and unit_text not in seen_units:
                    compound_units.append({
                        'match': match,
                        'unit_text': unit_text,
                        'start': match.start(),
                        'end': match.end()
                    })
                    seen_units.add(unit_text)
        
        return compound_units
    
    def _is_valid_compound_unit(self, unit_text: str) -> bool:
        """
        验证是否为有效的复合单位
        
        Args:
            unit_text: 单位文本
            
        Returns:
            bool: 是否为有效复合单位
        """
        # 简单验证：检查是否包含已知的单位组件
        known_unit_parts = ['L', 'l', 'g', 'kg', 'm', 'cm', 'mm', 'min', 'h', 'hr', 'kWh', 'MJ', 'kJ', 'W', 'kW']
        
        # 分解复合单位
        parts = re.split(r'[/²³\s]+', unit_text)
        
        # 至少有一个部分是已知单位
        return any(part in known_unit_parts for part in parts if part)
    
    def _build_compound_unit_examples(self, number_matches: List[re.Match], compound_units: List[Dict], pattern_count: int) -> List[str]:
        """
        构建复合单位模式的matched_examples
        
        Args:
            number_matches: 数值匹配列表
            compound_units: 复合单位列表
            pattern_count: 模式计数
            
        Returns:
            List[str]: 示例列表
        """
        examples = []
        
        # 显示前几个数值
        for i, match in enumerate(number_matches[:min(3, pattern_count)]):
            examples.append(match.group(0))
        
        # 显示复合单位
        if compound_units:
            unit_texts = [unit['unit_text'] for unit in compound_units[:2]]
            unique_units = list(dict.fromkeys(unit_texts))  # 去重保序
            if len(unique_units) == 1:
                examples.append(f"({unique_units[0]})")
            else:
                examples.append(f"({', '.join(unique_units)})")
        
        return examples[:3]
    
    def _calculate_qps_v2(self, pattern_count: int, text: str) -> float:
        """
        改进的quantitative_pattern_score计算（基于词密度）
        
        Args:
            pattern_count: 数值-单位对数量
            text: 文本内容
            
        Returns:
            float: QPS分数 (0-1)
        """
        if not text:
            return 0.0
        
        # 计算词数
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        # QPS = pattern_count / word_count
        qps = pattern_count / word_count
        
        # 限制在合理范围内（最大值1.0）
        return min(qps, 1.0)
    
    def _detect_table(self, text: str, metadata: Dict = None) -> bool:
        """
        检测是否为表格
        
        支持的表格类型：
        - table_section: 专家建议的表头重复分块（默认markdown格式）
        - table_keyvalue: 键值对格式（可选替代格式）
        
        已废弃类型（向后兼容）：
        - table_full: 已被table_section替代
        - table_row: 已被table_section替代
        """
        if metadata and 'chunk_type' in metadata:
            chunk_type = metadata['chunk_type']
            
            # 新的表格类型（推荐使用）
            if chunk_type in ['table_section', 'table_keyvalue']:
                return True
            
            # 废弃类型（向后兼容，但会记录警告）
            if chunk_type in ['table_full', 'table_row']:
                logger.warning(f"检测到废弃的表格类型: {chunk_type}，建议升级到table_section")
                return True
        
        # 没有元数据时默认为非表格
        return False
    
    def _empty_features(self) -> Dict[str, Any]:
        """返回空特征"""
        return {
            "pattern_count": 0,
            "quantitative_pattern_score": 0.0,
            "matched_examples": [],
            "contains_basis_tokens": False,
            "is_table": False
        }
    
    def validate_features(self, features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证特征数据的合理性"""
        warnings = []
        
        # 检查数值范围
        if features.get("quantitative_pattern_score", 0) > 1.0:
            warnings.append("QPS > 1.0, should be clamped to 1.0")
        
        # 检查低质量候选
        if (features.get("pattern_count", 0) == 0 and 
            features.get("quantitative_pattern_score", 0) < 0.02):
            warnings.append("Low quantitative evidence, consider Pivot")
        
        return len(warnings) == 0, warnings


# 全局实例
decision_features_calculator = DecisionFeaturesCalculator()


def calculate_decision_features(text: str, metadata: Dict = None) -> Dict[str, Any]:
    """
    便捷函数：计算决策特征
    
    Args:
        text: 文本内容
        metadata: 元数据
        
    Returns:
        Dict: 决策特征
    """
    return decision_features_calculator.calculate_features(text, metadata)


def validate_decision_features(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    便捷函数：验证决策特征
    
    Args:
        features: 特征字典
        
    Returns:
        Tuple: (是否有效, 警告列表)
    """
    return decision_features_calculator.validate_features(features)


if __name__ == "__main__":
    # 测试代码
    test_cases = [
        "Cooling Water | 5,000 L | per batch",
        "Energy demand description without numbers.",
        "Electricity consumption: 800 kWh per tonne of steel",
        "Table 1: Material inputs\nSteel: 1,200 kg\nWater: 3,500 L\nElectricity: 450 kWh"
    ]
    
    calculator = DecisionFeaturesCalculator()
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Text: {text}")
        features = calculator.calculate_features(text)
        print(f"Features: {features}")
        
        is_valid, warnings = calculator.validate_features(features)
        if warnings:
            print(f"Warnings: {warnings}")
