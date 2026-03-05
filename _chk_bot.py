import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# Status do servico
_, o, e = ssh.exec_command("systemctl status futebot --no-pager -l")
print("=== STATUS SERVICO ===")
print(o.read().decode())
print(e.read().decode())

# Ultimas 50 linhas do journal
_, o, e = ssh.exec_command("journalctl -u futebot --no-pager -n 50")
print("=== ULTIMAS 50 LINHAS DO LOG ===")
print(o.read().decode())

# Checar se o processo python ta rodando
_, o, _ = ssh.exec_command("ps aux | grep futebot | grep -v grep")
print("=== PROCESSOS ===")
print(o.read().decode())

ssh.close()
