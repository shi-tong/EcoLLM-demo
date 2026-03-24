"""
关键词建议器 - 基于两层关键词体系

功能：
1. 核心词库（Core）：必选的高频基础词汇
2. 扩展词库（Extended）：随机抽样的专业/长尾词汇
3. 自动生成搜索建议（5-8个关键词）

作者：AI Assistant
日期：2025-11-20
"""

import random
from typing import List, Dict
from enum import Enum


class LCICategory(str, Enum):
    """LCI数据类别"""
    FUNCTION_UNIT = "Function Unit"
    PRODUCT = "Product"
    RAW_MATERIAL = "Raw Material"
    PROCESS_ENERGY = "Process Energy"
    POST_PROCESSING_ENERGY = "Post-processing Energy"
    FEEDSTOCK_ENERGY = "Feedstock Energy"
    GAS = "Gas"
    COOLING_MEDIA = "Cooling Media"
    RECOVERED_MATERIAL = "Recovered Material"
    WASTE = "Waste"
    EMISSION = "Emission"


class KeywordSuggester:
    """关键词建议器"""
    
    # ============================================
    # SLM工艺关键词库（Phase 1）
    # ============================================
    
    KEYWORDS = {
        # Phase 1: Function Unit (Anchor)
        # 策略：通过Product反推Function Unit（同一物理对象，两种记录方式）
        # 搜索优先级：1. 生产描述语句 "We manufactured 10 parts..."
        #           2. Product相关内容作为上下文
        # 记录：Phase 1记录文本描述（定义层），Phase 2记录数值（计算层）
        LCICategory.FUNCTION_UNIT: {
            "core": [
                "manufactured",  # 生产动词（最高频）
                "part",          # 物理对象（最通用）
                "product",       # 产品（与Product类别关联）
                "kg"             # 单位（归一化基准常用）
            ],
            "extended": [
                "fabricated",       # 生产动词变体
                "printed",          # AM特有动词
                "produced",         # 生产动词变体
                "batch",            # 批次描述
                "component",        # 物理对象
                "specimen"          # 材料测试文献
            ]
        },
        
        # Phase 2: Output Flows
        LCICategory.PRODUCT: {
            "core": ["product", "part", "component"],
            "extended": ["output", "yield", "manufactured", "fabricated", "specimen"]
        },
        
        # Phase 3: Input Flows
        LCICategory.RAW_MATERIAL: {
            "core": ["powder", "material", "feedstock"],
            "extended": [
                "Ti6Al4V", "Ti-6Al-4V", "AlSi10Mg", "316L",
                "stainless steel", "aluminum alloy",
                "particle size", "virgin powder"
            ]
        },
        
        LCICategory.PROCESS_ENERGY: {
            "core": ["electricity", "kWh", "energy consumption"],
            "extended": [
                "power consumption", "MJ", "SEC", "specific energy consumption",
                "machine power", "printing energy", "build energy",
                "laser power", "heater power", "bed power"
            ]
        },
        
        LCICategory.POST_PROCESSING_ENERGY: {
            "core": ["heat treatment", "machining", "post-processing"],
            "extended": [
                "annealing", "stress relief", "furnace",
                "solution treatment", "aging",
                "CNC", "milling", "grinding", "cutting", "drilling",
                "surface finishing", "polishing", "blasting",
                "HIP", "hot isostatic pressing", "wire EDM", "EDM"
            ]
        },
        
        LCICategory.FEEDSTOCK_ENERGY: {
            "core": ["atomization", "powder production"],
            "extended": [
                "atomization energy", "powder manufacturing",
                "feedstock production", "feedstock energy",
                "gas atomization", "water atomization", "plasma atomization"
            ]
        },
        
        LCICategory.GAS: {
            "core": ["argon", "nitrogen", "gas"],
            "extended": [
                "inert gas", "shielding gas", "Ar", "N2",
                "gas consumption", "gas flow rate",
                "purge gas", "protective atmosphere",
                "compressed air"
            ]
        },
        
        LCICategory.COOLING_MEDIA: {
            "core": ["water", "coolant"],
            "extended": [
                "cooling water", "cutting fluid", "cooling liquid",
                "lubricant", "machining fluid", "coolant flow"
            ]
        },
        
        # Phase 3: Output Flows
        LCICategory.RECOVERED_MATERIAL: {
            "core": ["recovered powder", "recycled powder"],
            "extended": [
                "sieved powder", "reused powder",
                "unmelted powder", "loose powder",
                "surplus powder", "excess powder"
            ]
        },
        
        LCICategory.WASTE: {
            "core": ["waste", "scrap", "support"],
            "extended": [
                "support structure", "failed print", "rejected part",
                "machining waste", "trimmings", "offcuts",
                "powder waste", "contaminated powder",
                "condensate", "filter"
            ]
        },
        
        LCICategory.EMISSION: {
            "core": ["emission", "particulate"],
            "extended": [
                "VOC", "volatile organic compound",
                "particle", "fume", "smoke",
                "wastewater", "effluent", "off-gas",
                "air emission", "dust"
            ]
        }
    }
    
    def __init__(self, process_type: str = "SLM"):
        """
        初始化关键词建议器
        
        Args:
            process_type: 工艺类型（默认SLM，未来可扩展FDM/SLA等）
        """
        self.process_type = process_type
        self.history = []  # 历史使用记录
    
    def suggest_keywords(
        self,
        category: str,
        min_keywords: int = 5,
        max_keywords: int = 8,
        extended_count: int = None
    ) -> List[str]:
        """
        生成关键词建议
        
        Args:
            category: LCI类别（如 "Process Energy"）
            min_keywords: 最少关键词数量
            max_keywords: 最多关键词数量
            extended_count: 扩展词数量（None=随机1-2个）
        
        Returns:
            关键词列表
        """
        # 获取关键词库
        try:
            category_enum = LCICategory(category)
            keywords = self.KEYWORDS.get(category_enum)
        except ValueError:
            # 如果类别不存在，返回空列表
            return []
        
        if not keywords:
            return []
        
        core = keywords["core"]
        extended = keywords["extended"]
        
        # 核心词全部包含
        result = core.copy()
        
        # 随机抽取扩展词
        if extended_count is None:
            extended_count = random.randint(1, 2)
        
        # 确保不超过最大数量
        available_slots = max_keywords - len(core)
        extended_count = min(extended_count, available_slots, len(extended))
        
        if extended_count > 0:
            sampled_extended = random.sample(extended, k=extended_count)
            result.extend(sampled_extended)
        
        # 确保满足最小数量（如果可能）
        if len(result) < min_keywords and len(extended) > extended_count:
            additional_needed = min_keywords - len(result)
            remaining_extended = [kw for kw in extended if kw not in result]
            additional = random.sample(
                remaining_extended,
                k=min(additional_needed, len(remaining_extended))
            )
            result.extend(additional)
        
        # 记录到历史
        self.history.append({
            "category": category,
            "keywords": result.copy()
        })
        
        return result
    
    def get_all_categories(self) -> List[str]:
        """获取所有可用的类别"""
        return [cat.value for cat in LCICategory]
    
    def get_category_keywords(self, category: str) -> Dict[str, List[str]]:
        """
        获取指定类别的完整关键词库
        
        Args:
            category: LCI类别
        
        Returns:
            包含core和extended的字典
        """
        try:
            category_enum = LCICategory(category)
            return self.KEYWORDS.get(category_enum, {"core": [], "extended": []})
        except ValueError:
            return {"core": [], "extended": []}
    
    def check_diversity(self) -> Dict[str, float]:
        """
        检查历史使用的多样性
        
        Returns:
            关键词使用频率统计
        """
        if not self.history:
            return {}
        
        keyword_count = {}
        total_searches = len(self.history)
        
        for record in self.history:
            for kw in record["keywords"]:
                keyword_count[kw] = keyword_count.get(kw, 0) + 1
        
        # 计算频率
        keyword_freq = {
            kw: count / total_searches
            for kw, count in keyword_count.items()
        }
        
        return keyword_freq


# 全局实例
keyword_suggester = KeywordSuggester()
