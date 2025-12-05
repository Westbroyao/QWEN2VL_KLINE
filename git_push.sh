#!/usr/bin/env bash
# 一键 add + commit + push 脚本（支持切换 / 新建分支，支持 nothing to commit 也继续 push）

set -e

# 当前所在分支
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "当前所在分支：$CURRENT_BRANCH"

# 询问目标分支（默认用当前分支）
read -rp "请输入要推送的目标分支（直接回车使用当前分支：$CURRENT_BRANCH）: " TARGET_BRANCH

if [ -z "$TARGET_BRANCH" ]; then
  BRANCH="$CURRENT_BRANCH"
else
  BRANCH="$TARGET_BRANCH"
fi

# 如果当前分支不是目标分支，尝试切换
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
  echo
  echo "准备切换到分支：$BRANCH"

  # 判断本地是否已有这个分支
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo "本地已存在分支 $BRANCH，执行：git checkout $BRANCH"
    git checkout "$BRANCH"
  else
    echo "本地不存在分支 $BRANCH。"
    read -rp "是否新建分支 $BRANCH 并切换过去？[y/N] " CREATE_BRANCH
    if [[ "$CREATE_BRANCH" =~ ^[Yy]$ ]]; then
      echo "执行：git checkout -b $BRANCH"
      git checkout -b "$BRANCH"
    else
      echo "未切换分支，脚本结束。"
      exit 0
    fi
  fi
fi

echo
echo "当前目标分支：$BRANCH"
echo

# 如果命令行有参数，就把参数当作 commit message，否则交互输入
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