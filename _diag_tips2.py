import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# 1. Quando o servico foi parado/startado
_, o, _ = ssh.exec_command("journalctl -u futebot --no-pager --since '2026-03-01 22:00' --until '2026-03-02 14:00' | grep -i 'start\\|stop\\|signal\\|exited'")
print("HISTORICO SERVICO:")
print(o.read().decode())

# 2. Scanner rodou hoje?
_, o, _ = ssh.exec_command("journalctl -u futebot --no-pager --since '2026-03-02 06:30' --until '2026-03-02 08:00' | grep -i 'scanner\\|scan\\|tip\\|oportunidade'")
scan = o.read().decode()
print("SCANNER HOJE (06:30-08:00):")
print(scan if scan.strip() else "(vazio - NAO RODOU)")

# 3. Servico agora esta baixando stats - quando termina?
_, o, _ = ssh.exec_command("tail -5 /proc/$(pgrep -f 'gunicorn.*futebot' | head -1)/fd/1 2>/dev/null || journalctl -u futebot --no-pager -n 5")
print("\nULTIMAS LINHAS:")
print(o.read().decode())

# 4. Posso rodar scanner manualmente?
_, o, e = ssh.exec_command("cd /opt/futebot && ./venv/bin/python -c 'from pipeline.scanner import Scanner; from data.database import Database; s=Scanner(Database()); print(\"Scanner OK\")' 2>&1")
print("SCANNER IMPORT:")
print(o.read().decode())
err = e.read().decode()
if err:
    lines = [l for l in err.split('\n') if 'Warning' not in l and l.strip()]
    if lines:
        print("ERROS:", '\n'.join(lines[:10]))

ssh.close()
