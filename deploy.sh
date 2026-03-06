#!/usr/bin/env bash
# ──────────────────────────────────────────────
# deploy.sh — Envia código para o VPS e reinicia serviços
#
# Uso:
#   ./deploy.sh              → Deploy completo (sync + restart)
#   ./deploy.sh --sync-only  → Apenas sincronizar arquivos
#   ./deploy.sh --setup      → Primeira instalação (cria user, venv, serviços)
#
# Pré-requisitos:
#   - SSH configurado (ou senha será solicitada)
#   - rsync instalado localmente
#
# Dica: adicione sua chave SSH ao VPS para não digitar senha:
#   ssh-copy-id deploy@seu-servidor
# ──────────────────────────────────────────────

set -euo pipefail

# ── Configuração do VPS ──
VPS_HOST="${VPS_HOST:-seu-servidor}"
VPS_USER="${VPS_USER:-deploy}"
VPS_PORT="${VPS_PORT:-22}"
APP_DIR="${APP_DIR:-/opt/futebot}"
SERVICE_USER="${SERVICE_USER:-futebot}"

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERRO]${NC} $1"; exit 1; }

SSH_CMD="ssh -p ${VPS_PORT} ${VPS_USER}@${VPS_HOST}"

# ── Sincronizar arquivos ──
sync_files() {
    info "Sincronizando arquivos para ${VPS_HOST}:${APP_DIR}..."

    rsync -avz --delete \
        --exclude '.env' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.venv' \
        --exclude 'venv' \
        --exclude 'data/futebot.db' \
        --exclude 'data/futebot.db-wal' \
        --exclude 'data/futebot.db-shm' \
        --exclude 'data/models/*.json' \
        --exclude '.git' \
        -e "ssh -p ${VPS_PORT}" \
        . "${VPS_USER}@${VPS_HOST}:${APP_DIR}/"

    info "Arquivos sincronizados!"
}

# ── Reiniciar serviços ──
restart_services() {
    info "Reiniciando serviço no VPS..."

    ${SSH_CMD} << 'REMOTE'
        # Ajustar permissões
        chown -R futebot:futebot /opt/futebot

        # Instalar novas dependências (se houver)
        sudo -u futebot /opt/futebot/venv/bin/pip install -q -r /opt/futebot/requirements.txt

        # Reiniciar serviços
        systemctl restart futebot

        # Verificar status
        echo ""
        echo "=== Status do serviço ==="
        systemctl is-active futebot && echo "  ✅ futebot (bot+scheduler)" || echo "  ❌ futebot FALHOU"
        echo ""
REMOTE

    info "Deploy concluído!"
}

# ── Setup inicial (primeira vez) ──
setup_vps() {
    info "Executando setup inicial no VPS..."

    ${SSH_CMD} << REMOTE
        set -e

        # 1. Instalar dependências do sistema
        echo "📦 Instalando pacotes do sistema..."
        apt-get update -qq
        apt-get install -y -qq python3 python3-venv python3-pip

        # 2. Criar usuário de serviço (sem login)
        if ! id -u ${SERVICE_USER} &>/dev/null; then
            echo "👤 Criando usuário ${SERVICE_USER}..."
            useradd --system --home-dir ${APP_DIR} --shell /usr/sbin/nologin ${SERVICE_USER}
        fi

        # 3. Criar diretório
        mkdir -p ${APP_DIR}/data/models
        chown -R ${SERVICE_USER}:${SERVICE_USER} ${APP_DIR}

        # 4. Criar venv
        echo "🐍 Criando ambiente virtual..."
        sudo -u ${SERVICE_USER} python3 -m venv ${APP_DIR}/venv

        # 5. Instalar dependências Python
        echo "📥 Instalando dependências Python..."
        sudo -u ${SERVICE_USER} ${APP_DIR}/venv/bin/pip install -q --upgrade pip
        sudo -u ${SERVICE_USER} ${APP_DIR}/venv/bin/pip install -q -r ${APP_DIR}/requirements.txt

        # 6. Copiar serviço systemd
        cp ${APP_DIR}/deploy/futebot.service /etc/systemd/system/
        systemctl daemon-reload

        # 7. Habilitar serviço (inicia no boot)
        systemctl enable futebot

        echo ""
        echo "✅ Setup concluído!"
        echo ""
        echo "⚠️  PRÓXIMOS PASSOS MANUAIS:"
        echo "   1. Copie o .env para o VPS:"
        echo "      scp -P ${VPS_PORT} .env ${VPS_USER}@${VPS_HOST}:${APP_DIR}/.env"
        echo ""
        echo "   2. Inicie o serviço:"
        echo "      systemctl start futebot"
        echo ""
REMOTE

    info "Setup inicial concluído!"
}

# ── Main ──
case "${1:-}" in
    --setup)
        sync_files
        setup_vps
        ;;
    --sync-only)
        sync_files
        ;;
    *)
        sync_files
        restart_services
        ;;
esac
