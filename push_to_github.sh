#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "⚠️ Por favor, informe seu usuário do GitHub."
    echo "Uso: ./push_to_github.sh <seu-usuario-github>"
    echo "Exemplo: ./push_to_github.sh dyegosantos"
    exit 1
fi

GITHUB_USER="$1"
REPO_NAME="ia-dados-avancados"

echo "==> Configurando remote origin..."
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
git branch -M main

echo "==> Enviando código para o GitHub (branch main)..."
git push -u origin main

echo ""
echo "🎉 Repositório publicado com sucesso!"
echo "Acesse em: https://github.com/${GITHUB_USER}/${REPO_NAME}"
