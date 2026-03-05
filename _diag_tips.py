import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# 1. Hora
_, o, _ = ssh.exec_command("date '+%Y-%m-%d %H:%M %Z'")
print("HORA:", o.read().decode().strip())

# 2. Servico rodando?
_, o, _ = ssh.exec_command("systemctl is-active futebot")
print("SERVICO:", o.read().decode().strip())

# 3. Logs do futebot (ultimas 80 linhas)
_, o, _ = ssh.exec_command("journalctl -u futebot --no-pager -n 80 --since '2026-03-02 06:00'")
print("\nLOGS FUTEBOT:")
print(o.read().decode())

# 4. Erros recentes
_, o, _ = ssh.exec_command("journalctl -u futebot --no-pager -n 50 --since '2026-03-02 06:00' | grep -i 'erro\\|error\\|traceback\\|exception\\|falh'")
errs = o.read().decode()
print("ERROS:")
print(errs if errs.strip() else "(nenhum)")

# 5. Verificar se predictor carrega modelos per-league
_, o, e = ssh.exec_command("cd /opt/futebot && ./venv/bin/python -c 'from models.predictor import Predictor; from data.database import Database; p=Predictor(Database()); print(\"Predictor OK\"); print(\"Modelos globais:\", len(p._modelos))' 2>&1")
print("\nPREDICTOR:")
print(o.read().decode())
print(e.read().decode())

ssh.close()
