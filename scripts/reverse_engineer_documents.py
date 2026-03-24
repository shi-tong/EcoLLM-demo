"""
逆向工程：从LCI数据生成工艺文档 v3.1.2

v3.1.2修复（表格PDF转换问题）：
- 强制要求表格前后空行（TABLE FORMATTING）
- 修复Pandoc表格识别问题

v3.1.1修复（术语泄露+格式问题）：
- 禁止LCA/LCI术语泄露（FORBIDDEN TERMINOLOGY）
- 禁止markdown代码块包裹（OUTPUT FORMAT）
- 强化Complex的Batch Record：Raw Sensor不显示计算结果

v3.1改进（特征引导+专家建议版）：
- 用"风格特征"代替"具体例子"，避免模板化
- Base Prompt通用化（适配短中长文档）
- 9种文档类型：短(1500-2500词) + 中(3000-4500词) + 长(5000-7000词)
- 增加对比数据标记规则（避免混淆LCI真实数据）
- Complex长文档局部聚合（保持可追溯性）
- Batch Record的Raw Sensor Dumps（更自然的参数化）

支持的难度级别：
- simple: 清晰集中（易于快速提取）
- medium: 分散叙述（需读多个section）
- complex: 参数呈现（需计算得结果）

文档类型体系：
【Tier 1 - 短文档 1500-2500词】
- batch_production_record: 批次生产记录
- build_job_log: 设备作业日志
- material_traceability: 物料追溯报告
- quality_inspection: 质量检验报告
- process_narrative_memo: 工艺运行叙述备忘（高文本、少表格）

【Tier 2 - 中文档 3000-4500词】
- technical_process_report: 技术工艺报告
- environmental_assessment: 环境评估总结
- multi_build_analysis: 多批次对比分析

【Tier 3 - 长文档 5000-7000词】
- research_case_study: 研究案例分析
- sustainability_report: 可持续性报告章节
"""

import json
import os
from typing import List, Dict
from openai import OpenAI

class DocumentReverseEngineer:
    """
    从LCI数据逆向生成工艺文档
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        """
        初始化
        
        Args:
            api_key: DeepSeek API key
            base_url: API base URL
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def generate_document(
        self, 
        lci_data: Dict, 
        difficulty: str = "simple",
        document_type: str = "batch_production_record"
    ) -> str:
        """
        生成工艺文档
        
        Args:
            lci_data: LCI数据字典
                {
                    "process_name": "Ti6Al4V Femoral Stem Production",
                    "inputs": [
                        {"category": "Raw Material", "name": "Ti6Al4V Powder", "value": 20.83, "unit": "kg", "is_parent": True},
                        {"category": "Process Energy", "name": "Electricity - EOS M290", "value": 147.26, "unit": "kWh", "is_parent": True},
                        {"category": "Post-processing Energy", "name": "Heat treatment energy", "value": 2.5, "unit": "kWh", "is_parent": True},
                        {"category": "Feedstock Energy", "name": "Atomization energy", "value": 495, "unit": "MJ", "is_parent": True},
                        {"category": "Gas", "name": "Argon", "value": 10.5, "unit": "kg", "is_parent": True},
                        ...
                    ],
                    "outputs": [
                        {"category": "Product", "name": "Femoral Stems", "value": 1.77, "unit": "kg", "quantity": 20, "is_parent": True},
                        {"category": "Recovered Material", "name": "Recovered Ti6Al4V powder", "value": 3.2, "unit": "kg", "is_parent": True},
                        {"category": "Waste", "name": "Support structures", "value": 0.8, "unit": "kg", "is_parent": True},
                        ...
                    ],
                    "parameters": [
                        {"name": "Build Time", "value": 14.5, "unit": "h"},
                        {"name": "Laser Power", "value": 370, "unit": "W"},
                        ...
                    ]
                }
            difficulty: 难度等级 ("simple", "medium", "complex")
            document_type: 文档类型
                - "batch_production_record": 批量生产记录
                - "process_certification": 工艺认证报告
                - "material_traceability": 材料追溯报告
                - "build_job_log": 构建作业日志
        
        Returns:
            生成的文档文本（Markdown格式）
        """
        
        # 构建prompt
        prompt = self._build_prompt(lci_data, difficulty, document_type)
        
        # 调用LLM生成
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": self._get_system_prompt(difficulty, document_type)},
                {"role": "user", "content": prompt}
            ],
            temperature=1.3,  # 高温度以最大化多样性和创造性
            max_tokens=32000   # DeepSeek Reasoner支持更大的输出
        )
        
        document_text = response.choices[0].message.content
        return document_text
    
    def _get_system_prompt(self, difficulty: str, document_type: str) -> str:
        """
        获取系统提示词（v3.1: 特征引导+专家建议版）
        
        设计原则：
        - 用"风格特征"代替"具体例子"（避免模板化）
        - Base层通用化（适配所有长度）
        - Document Type用"typical characteristics"描述
        - 让Deepseek自由创作，而非套模板
        
        v3.1新增：
        - 对比数据标记规则
        - Complex长文档局部聚合
        - Batch Record的Raw Sensor Dumps
        """
        
        # ============================================================
        # Base Prompt: 通用原则（适配所有文档类型和长度）
        # ============================================================
        base_prompt = """You are creating documentation for an additive manufacturing process based on provided inventory data.

Your goal is to write authentic, domain-appropriate documents for the specified document type and audience. The document should read like something a real professional in this context would write for their actual work needs.

**GROUND TRUTH**
- Treat the provided inventory values (materials, energy, products, waste, gases, parameters) as the factual basis behind the document.
- It is important that the main resource flows (key materials, energy, gas, product and waste amounts) appear somewhere in the document with their correct magnitudes and units.
- You may add realistic contextual detail, and you may choose which supporting parameters to surface where they are most helpful, but you must not change or contradict the given values.

**DATA PLACEMENT AND SUMMARIES**
- Do not create a single bullet list, table, or paragraph that restates every numerical fact in one place. Let values appear in the sections where they naturally belong (e.g., material balance, energy analysis, gas use, KPIs, machine parameters).
- Executive summaries and introductory sections should focus on objectives, qualitative outcomes, and at most a few headline quantitative figures (for example total build time, approximate total energy use, or approximate material utilization).
- In high-level summaries it is acceptable to round values to realistic figures (e.g., "about 0.49 kg of powder", "around 11 kWh of electricity"), while keeping exact values in tables or detailed sections.

**ROLE OF THE DOCUMENT**
- The primary purpose of the document is its stated business purpose (e.g., batch record, technical report, case study), not "presenting LCI data".
- Inventory data should appear where it naturally belongs for that purpose: as part of operations, methods, results, or traceability, depending on the document type.

**STYLE AND TONE**
- Use a professional tone consistent with the document type and audience.
- Structure the document in a way that would be natural for practitioners in this domain, not optimized for machine extraction.
- Include sufficient context so that a human reader unfamiliar with the specific run can still understand what was done.

**DOCUMENT LENGTH AND DEPTH**
- For short, operational document types such as batch production records, build job logs, material traceability reports and quality inspection reports, aim for roughly 2000–3000 words.
- For medium-length analytical documents such as technical process reports, environmental assessments, multi-build analyses, process development reports, manufacturing feasibility studies and process characterization studies, aim for roughly 3500–5000 words.
- For in-depth case studies and sustainability chapters (e.g., research case studies and sustainability report chapters), aim for roughly 6000–8000 words.

These are soft targets: you do not need to hit an exact word count, but the document should be substantial enough to fully serve its professional purpose rather than a brief summary.

**CONTENT RICHNESS**
- Beyond listing numerical values, include appropriate narrative context: background or motivation where relevant, concise process descriptions, and interpretation of what the results imply for operations or engineering decisions.
- Where it is natural for the document type, briefly mention limitations, assumptions, or next steps in your own words, rather than following a fixed template.

**INTERPRETATION STYLE**
- When you explain how values are calculated or related, prefer short, natural-language descriptions over long, step-by-step algebraic formulas or code-like expressions.
- Do not criticize, "correct", or override the given numeric values. Treat them as the authoritative recorded measurements for this document, even when they could be recombined or re-derived from each other.
- Avoid meta-comments such as "the provided data lists...", "this dataset shows...", or "the input data appears inconsistent". Write as if you are the engineer or analyst reporting your own measurements and calculations inside the document.

**FORBIDDEN TERMINOLOGY**
- Do NOT use life cycle assessment jargon (e.g., "LCA", "LCI", "life cycle inventory", "functional unit", "impact assessment").
- These documents are operational, technical, or managerial records, not explicit sustainability studies.

**OUTPUT FORMAT**
- Output ONLY the document content in plain Markdown.
- Do NOT wrap the output in markdown code fences.
- Start directly with the document title as a level-1 heading (# Title).

**TABLES AND PDF CONVERSION**
- When you include tables, use standard Markdown table syntax with header and separator rows.
- Always include a blank line before and after each table; this is important for the downstream Markdown-to-PDF conversion pipeline.
- Do not nest tables or mix table rows with list markers or headings on the same line.
"""
        
        # ============================================================
        # Difficulty Instructions: 数据呈现方式（控制表格/文本比例与显性/隐含程度）
        # ============================================================
        difficulty_instructions = {
            "simple": """
The document should behave like a clear, practical record where most quantitative information is presented through tables.

- Use several focused tables grouped by topic or process stage (e.g., material inputs, energy use, outputs), rather than a single table that tries to list every possible item.
- Tables should directly list values that would realistically be recorded in logs or reports (materials, energy, waste, key outputs), so that a reader can see the numbers without doing extra calculations.
- Narrative text should mainly tie the tables to the workflow and explain what they refer to; it may repeat or highlight important values, but should not be the only place where key quantities appear.

Overall, favor straightforward, table-centered presentation of data while still writing as a real practitioner would for this document type.
""",

            "medium": """
The document should combine tables and narrative text so that neither alone contains the full picture.

- Some important quantities should appear in tables (for structured reference), while others only appear inside paragraphs that describe activities, operating conditions, or simple calculations.
- It is natural to show how certain totals, intensities, or ratios are derived from underlying values, with at least a few such relationships explained explicitly in the text.
- Avoid designing a single "master paragraph" or table that exposes the entire material and energy balance; instead, let different parts of the balance appear in the sections where they are most naturally discussed.
- To reconstruct the complete resource inventory, a knowledgeable reader will need to read multiple sections and cross-reference both tables and prose.

Overall, favor a realistic mix of tables and narrative explanation, with modest but non-trivial reasoning needed to connect all the data.
""",

            "complex": """
The document should read like an in-depth technical or analytical report where many key quantities are embedded in narrative, parameters, and relationships.

- Emphasize parameters, intensities, rates, ratios, and operating conditions described in the text; final totals may be implied or scattered rather than always listed in one place.
- Use tables selectively for representative parameters, partial results, or local summaries, but avoid relying on a single exhaustive inventory table that lists every input and output flow explicitly.
- Clearly describe how quantities relate to each other (e.g., how power, time, and throughput combine), including major assumptions and their implications for performance.
- It is not necessary for any single section to expose every important flow explicitly; expect expert readers to cross-reference multiple sections when needed.
- Do not try to help the reader by repeating all major numeric values together in a recap paragraph; keep detailed breakdowns localized to where they are first introduced.
- A reader should need to interpret narrative descriptions, parameter values, and occasional tables together—often performing their own calculations—to reconstruct the full set of inventory data.

Overall, favor depth and reasoning-centered reporting, where understanding the process requires reading, connecting information across sections, and using the given relationships.
"""
        }
        
        # ============================================================
        # Document Type Instructions: 特征描述（按文档用途和受众分层）
        # ============================================================
        document_type_instructions = {
            # ========== Tier 1: 短文档 (2000-3000 words) ==========
            "batch_production_record": """
This document is a batch production record for a specific manufacturing batch. It is written mainly for production supervisors and quality staff to trace what was done, when it was done, and with which materials and equipment. The tone is factual and concise, and the organization typically follows the production flow or chronological order, including preparation, execution, and post-processing. Resource usage such as material quantities, gas consumption, machine run durations, and waste handling naturally appears while describing these concrete steps and checkpoints, rather than as a standalone data report.
""",

            "build_job_log": """
This document is a build job log focused on the operation of a particular machine or build job. It is written for machine operators, maintenance personnel, and process engineers. The emphasis is on equipment status, events, alarms, sensor readings, and interventions over time. The structure often resembles time-ordered entries or event-based sections. Resource-related information such as power draw, gas supply, or consumables appears as part of recording how the machine was set up, monitored, and shut down, not as the central topic. For medium and complex difficulty, prefer embedding more data in narrative descriptions of events rather than relying heavily on tables.
""",

            "material_traceability": """
This document is a material traceability report used for quality assurance and compliance. It is written for QA teams, supply chain managers, and auditors who need to follow materials from receipt through processing to final disposition. The focus is on lot numbers, quantities, locations, and transformations of materials, maintaining a clear chain of custody. Resource quantities naturally appear when recording receiving records, in-process movements, blending or splitting of lots, and final allocation to products or waste streams.
""",

            "quality_inspection": """
This document is a quality inspection report that records how parts or batches were inspected and whether they met specified criteria. It is written for quality control staff, customers, and sometimes certification bodies. The emphasis is on inspection procedures, sampling plans, measurement results, pass/fail decisions, and references to specifications. Resource and process data may appear where they are relevant to the inspection context, but the main focus is on conformity assessment rather than resource accounting.
""",

            "process_narrative_memo": """
This document is an internal narrative memo or run log written by a process engineer or senior operator after completing a manufacturing run. It is intended for colleagues, supervisors, or future reference. The writing style is prose-heavy: almost all information appears in paragraphs or short bullet lists, with minimal or no tables. Quantitative data such as material usage, energy consumption, gas supply, and waste handling are embedded naturally in sentences describing what was done and observed. The structure follows the chronological flow of the run or the logical sequence of setup, execution, and wrap-up, rather than a formal report template.
""",

            # ========== Tier 2: 中文档 (3500-5000 words) ==========
            "technical_process_report": """
This document is a technical process report describing a manufacturing process in depth for engineers and technical managers. It typically includes background, process description, methodology, results, and technical discussion. The writing balances narrative explanation with detailed technical data such as parameters, settings, and measured outcomes. Resource-related information appears as part of describing how the process is configured and how it performed, rather than as a separate inventory chapter. For medium and complex difficulty, lean toward prose-based explanations and use tables sparingly for key summaries only.
""",

            "environmental_assessment": """
This document is an environmental assessment-style process summary aimed at environmental managers or operational teams interested in resource use and emissions, without using formal life cycle assessment terminology. It highlights resource consumption, waste generation, and operational practices that influence environmental performance. Data about materials, energy, and waste appears in the context of describing process stages and their implications, and may be compared qualitatively or quantitatively to previous periods or generic benchmarks.
""",

            "multi_build_analysis": """
This document is a multi-build comparative analysis that looks across several production runs or scenarios. It is written for process engineers and production management to understand variability and trends. The organization typically contrasts different builds or conditions, discussing performance metrics, stability, and observed differences. Resource and output data are used to support comparisons across builds, appearing where they help explain patterns, deviations, or improvement opportunities.
""",

            # ========== Tier 3: 长文档 (6000-8000 words) ==========
            "research_case_study": """
This document is a research-style case study intended for a technical or academic audience. It usually follows a structured flow from background and objectives through methodology, results, and discussion to conclusions. The emphasis is on transparency of methods, reproducibility, and careful interpretation of findings. Resource and process data are embedded within the methodological description and the presentation of results, and may be revisited in the discussion when interpreting process behavior or performance.
""",

            "sustainability_report": """
This document is a chapter of a broader sustainability or corporate responsibility report. It is written for a mixed audience that can include internal stakeholders, investors, and external readers. The writing combines narrative framing with selected quantitative indicators, often connecting operational practices to broader sustainability goals. Resource and waste data from the process appear as part of illustrating performance, efficiency, and improvement efforts, alongside qualitative explanations and contextual information.
""",

            # ========== 新增文档类型（v5.0）：文本丰富，训练检索能力 ==========
            "process_development_report": """
This document is a process development report describing how a manufacturing process was developed and refined over time. It is written for process engineers, R&D teams, and technical leadership. The narrative typically covers stages of experimentation, parameter exploration, design choices, and lessons learned. Resource and performance data are used to illustrate why certain configurations were chosen or rejected, appearing naturally in the story of development rather than in a single consolidated inventory section.
""",

            "manufacturing_feasibility_study": """
This document is a manufacturing feasibility study prepared to support decisions about adopting or scaling a process. It is written for decision-makers, engineers, and operations planners. The content considers technical requirements, resource needs, infrastructure implications, risks, and practical constraints. Resource and capacity data are woven into the analysis of feasibility, trade-offs, and recommendations, rather than being presented in isolation.
""",

            "process_characterization_study": """
This document is a process characterization study aimed at understanding how a process behaves under different conditions. It is written for process engineers and researchers. The structure often reflects experimental or observational campaigns, analyzing relationships between inputs, parameters, and outputs. Resource and process data appear throughout the description of test conditions and results, supporting analysis of sensitivities, dependencies, and stable operating windows.
"""
        }
        
        
        # ============================================================
        # 组装最终Prompt
        # ============================================================
        final_prompt = base_prompt + \
                      difficulty_instructions.get(difficulty, difficulty_instructions["simple"]) + \
                      document_type_instructions.get(document_type, document_type_instructions["batch_production_record"])
        
        return final_prompt
    
    def _build_prompt(self, lci_data: Dict, difficulty: str, document_type: str) -> str:
        """构建用户prompt（v1.5: 支持relations）"""
        
        # 提取数据
        process_name = lci_data.get("process_name", "AM Process")
        inputs = lci_data.get("inputs", [])
        outputs = lci_data.get("outputs", [])
        parameters = lci_data.get("parameters", [])
        relations = lci_data.get("relations", [])
        
        # 构建清晰的数据列表
        data_items = []
        
        # 构建relations映射（flow名称 -> relation信息）
        relations_map = {}
        if relations:
            for rel in relations:
                flow_key = rel.get("flow", "")
                relations_map[flow_key] = rel
        
        for inp in inputs:
            flow_key = f"{inp['category']}: {inp['name']}"
            relation = relations_map.get(flow_key)
            
            if relation:
                # 有relation的flow，标注为CALCULABLE并包含计算说明
                calc_desc = relation.get("calculation", "")
                params_used = relation.get("parameters_used", [])
                data_items.append(f"[CALCULABLE] {inp['name']} ({inp['category']}): {inp['value']} {inp['unit']}")
                if difficulty == "complex":
                    data_items.append(f"  → Calculation: {calc_desc}")
                    data_items.append(f"  → Uses parameters: {', '.join(params_used)}")
            elif inp.get("child_nodes"):
                # 旧版child_nodes格式兼容
                child_str = ", ".join([f"{c['name']}: {c['value']} {c['unit']}" for c in inp['child_nodes']])
                data_items.append(f"[CALCULABLE] {inp['name']} ({inp['category']}): parameters are {child_str}, final value is {inp['value']} {inp['unit']}")
            else:
                # 标注为直接陈述的flow
                data_items.append(f"[DIRECT] {inp['name']} ({inp['category']}): {inp['value']} {inp['unit']}")
        
        for out in outputs:
            qty = f", {out['quantity']} units" if "quantity" in out else ""
            flow_key = f"{out['category']}: {out['name']}"
            relation = relations_map.get(flow_key)
            
            if relation:
                # 有relation的flow
                calc_desc = relation.get("calculation", "")
                params_used = relation.get("parameters_used", [])
                data_items.append(f"[CALCULABLE] {out['name']} ({out['category']}): {out['value']} {out['unit']}{qty}")
                if difficulty == "complex":
                    data_items.append(f"  → Calculation: {calc_desc}")
                    data_items.append(f"  → Uses parameters: {', '.join(params_used)}")
            elif out.get("child_nodes"):
                # 旧版child_nodes格式兼容
                child_str = ", ".join([f"{c['name']}: {c['value']} {c['unit']}" for c in out['child_nodes']])
                data_items.append(f"[CALCULABLE] {out['name']} ({out['category']}): parameters are {child_str}, final value is {out['value']} {out['unit']}{qty}")
            else:
                data_items.append(f"[DIRECT] {out['name']} ({out['category']}): {out['value']} {out['unit']}{qty}")
        
        if parameters:
            for param in parameters:
                data_items.append(f"[CONTEXT] {param['name']}: {param['value']} {param['unit']}")
        
        # 构建prompt说明
        notes = [
            "- [DIRECT] = state the final value directly",
            "- [CALCULABLE] = flow calculated from parameters",
            "- [CONTEXT] = additional process parameters for context"
        ]
        
        if difficulty == "complex" and relations:
            notes.append("\n**For Complex documents**: Present the underlying parameters for [CALCULABLE] flows rather than the final calculated value, and describe key relationships in concise prose rather than long algebraic derivations.")
        elif difficulty == "medium" and relations:
            notes.append("\n**Tip for Medium documents**:")
            notes.append("- You may briefly mention in words how some totals or ratios are obtained from underlying values, but avoid extended step-by-step calculations.")
        
        notes.append("\n**Integration guidance**:")
        notes.append("- Do not simply copy the 'Data to document' list into the report as a single block.")
        notes.append("- Integrate these values into the sections where they naturally belong (for example material balance, energy analysis, gas consumption, results, or KPI tables).")
        notes.append("- It is acceptable if some contextual parameters are only mentioned once in a focused section, rather than being repeated many times across the document.")
        
        prompt = f"""Process: {process_name}

Data to document:
{chr(10).join('- ' + item for item in data_items)}

Note: 
{chr(10).join(notes)}

Write a formal {document_type.replace('_', ' ')} using tables and narrative paragraphs. Start with a clear document title using # heading. Use proper Markdown formatting (headings, bold text, tables, horizontal rules). Return only the Markdown document.
"""
        
        return prompt
    
    def batch_generate(
        self,
        lci_data_list: List[Dict],
        output_dir: str = "dataset/documents"
    ):
        """
        批量生成文档
        
        Args:
            lci_data_list: LCI数据列表
            output_dir: 输出目录
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 难度和文档类型组合（v2.0: Tier 1文档类型）
        difficulty_levels = ["simple", "medium", "complex"]
        document_types = [
            "batch_production_record",
            "material_traceability",
            "build_job_log"
        ]
        
        generated_count = 0
        
        for i, lci_data in enumerate(lci_data_list):
            # 为每个LCI数据生成不同难度和类型的文档
            for difficulty in difficulty_levels:
                for doc_type in document_types:
                    try:
                        print(f"🔄 生成文档 {generated_count + 1}: {lci_data['process_name']} - {difficulty} - {doc_type}")
                        
                        document_text = self.generate_document(
                            lci_data=lci_data,
                            difficulty=difficulty,
                            document_type=doc_type
                        )
                        
                        # 保存文档
                        filename = f"doc_{i+1}_{difficulty}_{doc_type}.md"
                        filepath = os.path.join(output_dir, filename)
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(document_text)
                        
                        # 保存元数据
                        metadata = {
                            "source_lci": lci_data,
                            "difficulty": difficulty,
                            "document_type": doc_type,
                            "filename": filename
                        }
                        
                        metadata_file = filepath.replace('.md', '_metadata.json')
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        generated_count += 1
                        print(f"✅ 已保存: {filepath}")
                        
                    except Exception as e:
                        print(f"❌ 生成失败: {e}")
        
        print(f"\n🎉 批量生成完成！共生成 {generated_count} 个文档")


def load_lci_from_literature(literature_file: str) -> List[Dict]:
    """
    从文献中提取LCI数据
    
    Args:
        literature_file: 文献文件路径（JSON格式）
            格式示例：
            {
                "papers": [
                    {
                        "title": "...",
                        "lci_data": {
                            "process_name": "...",
                            "inputs": [...],
                            "outputs": [...],
                            "parameters": [...]
                        }
                    }
                ]
            }
    
    Returns:
        LCI数据列表
    """
    with open(literature_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lci_list = []
    for paper in data.get("papers", []):
        lci_data = paper.get("lci_data")
        if lci_data:
            lci_list.append(lci_data)
    
    return lci_list


# ============================================================
# 示例使用
# ============================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='从LCI数据生成工艺文档')
    parser.add_argument('--input', type=str, required=True, help='输入的JSON文件路径')
    parser.add_argument('--difficulty', type=str, default='simple', choices=['simple', 'medium', 'complex'], help='难度级别')
    parser.add_argument('--document-type', type=str, default='batch_production_record', help='文档类型')
    args = parser.parse_args()
    
    # 检查API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 DEEPSEEK_API_KEY")
        print("   请运行: export DEEPSEEK_API_KEY='your_key'")
        sys.exit(1)
    
    # 从文件加载LCI数据
    print(f"📖 从文件加载LCI数据: {args.input}")
    lci_list = load_lci_from_literature(args.input)
    
    if len(lci_list) == 0:
        print("❌ 错误: 没有找到LCI数据")
        print(f"   请确保 {args.input} 文件存在且格式正确")
        sys.exit(1)
    
    print(f"✅ 找到 {len(lci_list)} 个LCI数据，使用第一个")
    lci_to_use = lci_list[0]
    
    # 初始化生成器
    engineer = DocumentReverseEngineer(api_key=api_key)
    
    # 生成单个文档示例
    print("🚀 生成示例文档...")
    doc = engineer.generate_document(
        lci_data=lci_to_use,
        difficulty=args.difficulty,
        document_type=args.document_type
    )
    
    # 保存到文件
    output_dir = "dataset/documents"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"test_{args.difficulty}_{args.document_type}.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(doc)
    
    print("\n" + "="*60)
    print(f"✅ 文档已保存到: {output_file}")
    print("\n文档内容预览（前500字符）:")
    print(doc[:500] + "...")
    print(f"\n📊 文档总长度: {len(doc)} 字符")
    
    # 批量生成
    # engineer.batch_generate([example_lci])
