#!/bin/bash
# 云服务器环境配置脚本

echo "=== 1. 克隆 LLaMA Factory ==="
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

echo "=== 2. 安装依赖 ==="
pip install -e ".[torch,metrics]" --no-build-isolation

echo "=== 3. 复制数据文件 ==="
# 假设你已经上传了 lca_data.jsonl 到当前目录
cp ../lca_data.jsonl data/

echo "=== 4. 更新 dataset_info.json ==="
# 在 dataset_info.json 开头添加 lca_data 配置
python3 << 'EOF'
import json

config_path = "data/dataset_info.json"
with open(config_path, 'r') as f:
    config = json.load(f)

lca_config = {
    "lca_data": {
        "file_name": "lca_data.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
            "observation_tag": "observation",
            "function_tag": "function_call"
        }
    }
}

# 合并配置
config = {**lca_config, **config}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("✅ dataset_info.json 已更新")
EOF

echo "=== 5. 复制训练配置 ==="
cp ../train_config.yaml .

echo "=== 环境配置完成 ==="
echo "运行训练: llamafactory-cli train train_config.yaml"
