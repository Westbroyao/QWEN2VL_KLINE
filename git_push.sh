#!/usr/bin/env bash
# 一键 add + commit + push 脚本（支持 nothing to commit 也继续 push）

set -e

# 取得当前分支名（比如 main / dev）
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "当前分支：$BRANCH"
echo

# 如果命令行有参数，就把参数当作 commit message
# 否则就交互式输入
if [ $# -gt 0 ]; then
  COMMIT_MSG="$*"
else
  read -rp "请输入 commit message: " COMMIT_MSG
fi

if [ -z "$COMMIT_MSG" ]; then
  echo "commit message 不能为空，已退出。"
  exit 1
fi

echo
echo "===> 当前 git 状态："
git status
echo

read -rp "确认要 git add . 并推送到 origin/$BRANCH 吗？[y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "已取消。"
  exit 0
fi

echo
echo "===> git add ."
git add .

# 检查暂存区是否有改动
if git diff --cached --quiet; then
  echo "没有新的改动需要提交，跳过 commit，直接 push 到 origin/$BRANCH ..."
else
  echo "===> git commit -m \"$COMMIT_MSG\""
  git commit -m "$COMMIT_MSG"
fi

echo "===> git push origin $BRANCH"
git push origin "$BRANCH"

echo
echo "✅ 已推送到 origin/$BRANCH"