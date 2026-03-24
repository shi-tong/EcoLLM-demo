#!/bin/bash

# LCA-LLM 专家标注工作台启动脚本

echo "👨‍🔬 Starting LCA-LLM Expert Annotation Workbench..."

# 切换到项目目录
cd "$(dirname "$0")/.."

# 激活虚拟环境
if [ -d "lcaLLM" ]; then
    source lcaLLM/bin/activate
    echo "✅ 虚拟环境已激活"
else
    echo "❌ 未找到虚拟环境目录 lcaLLM"
    exit 1
fi

# 检查后端服务状态
echo "🔍 检查后端服务状态..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ 后端服务运行正常"
else
    echo "❌ 后端服务未运行，请先启动后端服务"
    echo "💡 运行: ./start_services.sh"
    exit 1
fi

# 检查关键API端点
echo "🔍 检查工具API端点..."
for endpoint in "process-document" "search-document" "define-lca-scope" "record-process-flow" "session-summary/test"; do
    if curl -s "http://localhost:8000/tools/$endpoint" > /dev/null; then
        echo "✅ /tools/$endpoint - 可用"
    else
        echo "⚠️ /tools/$endpoint - 可能不可用"
    fi
done

echo "🚀 启动专家标注工作台..."
echo "📍 访问地址: http://localhost:8504"
echo "⏹️  停止: Ctrl+C"
echo ""

# 启动Streamlit应用
export STREAMLIT_SERVER_PORT=8504
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

streamlit run scripts/expert_annotation_workbench.py \
    --server.port 8504 \
    --server.address 0.0.0.0 \
    --theme.base light \
    --theme.primaryColor "#1f77b4" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6"
