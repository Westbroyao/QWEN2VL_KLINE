#!/usr/bin/env bash
# 简单的 Git 全局配置脚本
# 用法：bash git_config_setup.sh

set -e

echo "=== Git 全局配置脚本 ==="

# 交互式输入
read -rp "请输入你的 Git 用户名（例如：Westbroyao）: " GIT_NAME
read -rp "请输入你的 Git 邮箱（用于 GitHub，例：xxx@gmail.com）: " GIT_EMAIL

echo
echo "设置全局 user.name / user.email ..."
git config --global user.name  "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

echo "设置默认主分支名称为 main ..."
git config --global init.defaultBranch main

echo "开启彩色输出 ..."
git config --global color.ui auto

echo "设置默认使用合并（而不是 rebase）方式 pull ..."
git config --global pull.rebase false

# 可选：缓存凭据一段时间（15 分钟），方便频繁 push
# 不想用可以注释掉这一行
echo "设置凭据缓存（15 分钟）..."
git config --global credential.helper 'cache --timeout=900'

echo
echo "当前 git 全局配置如下："
echo "--------------------------------"
git config --global --list
echo "--------------------------------"
echo "Git 配置完成 ✅"