#!/bin/bash
# 用当前工作区内容完全替换远程 Edit-Banana 仓库，清空历史，保留 Star。
# 使用前请阅读 REPLACE_REPO.md，确认 .gitignore 已排除敏感文件。
set -e
cd "$(dirname "$0")"
REPO_URL="git@github.com:BIT-DataLab/Edit-Banana.git"

echo "=== 1. 确保本地为 Git 仓库 ==="
if ! git rev-parse --git-dir &>/dev/null; then
  echo "  当前目录不是 Git 仓库，执行 git init ..."
  git init
  echo "  [OK] 已初始化"
else
  echo "  [OK] 已是 Git 仓库"
fi

echo ""
echo "=== 2. 检查敏感文件是否被忽略 ==="
if git check-ignore -q config/config.yaml 2>/dev/null; then
  echo "  [OK] config/config.yaml 已被 .gitignore 忽略"
else
  echo "  [WARN] config/config.yaml 未被忽略，请检查 .gitignore"
  exit 1
fi
if git check-ignore -q .env 2>/dev/null; then
  echo "  [OK] .env 已被忽略"
else
  echo "  [WARN] .env 未被忽略，请检查 .gitignore"
  exit 1
fi

echo ""
echo "=== 3. 设置远程为 Edit-Banana (SSH) ==="
if ! git remote get-url origin &>/dev/null; then
  git remote add origin "$REPO_URL"
else
  git remote set-url origin "$REPO_URL"
fi
git remote -v

echo ""
echo "=== 4. 创建无历史分支并提交当前内容 ==="
git checkout --orphan new_main 2>/dev/null || true
git rm -rf --cached . 2>/dev/null || true
git add .
echo "  即将提交的文件："
git status --short
read -p "  确认无敏感文件后按 Enter 继续，Ctrl+C 取消..."
git commit -m "Initial commit: algorithm pipeline only (Image to DrawIO)"

echo ""
echo "=== 5. 覆盖远程 main ==="
git branch -D main 2>/dev/null || true
git branch -m main
echo "  即将执行: git push -f origin main"
read -p "  确认后按 Enter 执行，Ctrl+C 取消..."
git push -f origin main

echo ""
echo "=== 完成 ==="
echo "  仓库已替换，历史已清空，Star/Forks 等不受影响。"
echo "  https://github.com/BIT-DataLab/Edit-Banana"
