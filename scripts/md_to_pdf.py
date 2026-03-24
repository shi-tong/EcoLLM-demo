#!/usr/bin/env python3
"""
将Markdown文档转换为PDF，确保表格边框正确显示
"""

import subprocess
import sys
from pathlib import Path

def md_to_pdf(md_file: str, output_pdf: str = None, css_file: str = None):
    """
    将Markdown转换为PDF
    
    Args:
        md_file: Markdown文件路径
        output_pdf: 输出PDF路径（可选，默认为同名.pdf）
        css_file: CSS样式文件路径（可选）
    """
    md_path = Path(md_file)
    if not md_path.exists():
        print(f"错误: 文件不存在 {md_file}")
        return False
    
    # 确定输出路径
    if output_pdf is None:
        output_pdf = md_path.with_suffix('.pdf')
    
    # 确定CSS路径
    if css_file is None:
        css_file = Path(__file__).parent.parent / "md2pdf_fixed.css"
    
    print(f"转换: {md_file} -> {output_pdf}")
    print(f"使用CSS: {css_file}")
    
    try:
        # 方法1: 使用pandoc + weasyprint命令行
        # 先转HTML
        temp_html = md_path.with_suffix('.temp.html')
        
        pandoc_cmd = [
            'pandoc',
            str(md_file),
            '--embed-resources',
            '--standalone',
            f'--css={css_file}',
            '-t', 'html',
            '-o', str(temp_html)
        ]
        
        print(f"执行: {' '.join(pandoc_cmd)}")
        result = subprocess.run(pandoc_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Pandoc错误: {result.stderr}")
            return False
        
        # 检查是否安装了weasyprint
        try:
            import weasyprint
            # 使用Python weasyprint库
            print("使用Python weasyprint库转换...")
            html_doc = weasyprint.HTML(filename=str(temp_html))
            html_doc.write_pdf(str(output_pdf))
            print(f"✅ 成功生成PDF: {output_pdf}")
            
            # 清理临时文件
            if temp_html.exists():
                temp_html.unlink()
            return True
            
        except ImportError:
            print("\n❌ 错误: 未安装weasyprint")
            print("\n请安装weasyprint:")
            print("  pip install weasyprint")
            print("\n或使用系统包管理器:")
            print("  sudo apt install python3-weasyprint")
            
            # 不清理临时文件，方便调试
            print(f"\n临时HTML文件保留在: {temp_html}")
            return False
        
        except Exception as e:
            print(f"\n❌ WeasyPrint转换错误: {e}")
            print(f"临时HTML文件: {temp_html}")
            return False
            
    except Exception as e:
        print(f"转换失败: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python md_to_pdf.py <markdown_file> [output_pdf] [css_file]")
        print("\n示例:")
        print("  python md_to_pdf.py document.md")
        print("  python md_to_pdf.py document.md output.pdf")
        print("  python md_to_pdf.py document.md output.pdf custom.css")
        sys.exit(1)
    
    md_file = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    css_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    success = md_to_pdf(md_file, output_pdf, css_file)
    sys.exit(0 if success else 1)
