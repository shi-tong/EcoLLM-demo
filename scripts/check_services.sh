#!/bin/bash

# LCA-LLM服务状态检查脚本
# 用于查看服务运行状态

echo "=========================================="
echo "LCA-LLM 服务状态检查"
echo "=========================================="
echo ""

# 检查后端
echo "1️⃣ 后端服务 (端口 8000):"
BACKEND_PID=$(pgrep -f "uvicorn backend.app:app")
if [ -n "$BACKEND_PID" ]; then
    echo "   ✅ 运行中 (PID: $BACKEND_PID)"
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✅ API响应正常"
    else
        echo "   ⚠️  API无响应"
    fi
else
    echo "   ❌ 未运行"
fi

# 检查前端
echo ""
echo "2️⃣ 前端工作台 (端口 8504):"
STREAMLIT_PID=$(pgrep -f "streamlit run scripts/expert_annotation_workbench.py")
if [ -n "$STREAMLIT_PID" ]; then
    echo "   ✅ 运行中 (PID: $STREAMLIT_PID)"
    if curl -s http://localhost:8504 > /dev/null 2>&1; then
        echo "   ✅ 页面响应正常"
    else
        echo "   ⚠️  页面无响应（可能正在启动）"
    fi
else
    echo "   ❌ 未运行"
fi

# 检查MongoDB
echo ""
echo "3️⃣ MongoDB (端口 27017):"
if pgrep -x mongod > /dev/null; then
    echo "   ✅ 运行中"
    if mongosh --eval "db.adminCommand('ping')" --quiet > /dev/null 2>&1; then
        echo "   ✅ 连接正常"
    else
        echo "   ⚠️  连接失败"
    fi
else
    echo "   ❌ 未运行"
fi

echo ""
echo "=========================================="
echo "📝 日志位置:"
echo "   后端: /tmp/backend.log"
echo "   前端: /tmp/streamlit.log"
echo ""
echo "🔧 快速命令:"
echo "   查看后端日志: tail -f /tmp/backend.log"
echo "   查看前端日志: tail -f /tmp/streamlit.log"
echo "   重启所有服务: ./scripts/restart_services.sh"
echo "   停止所有服务: ./scripts/stop_services.sh"
echo "=========================================="
echo ""

