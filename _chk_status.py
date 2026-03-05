import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# Hora atual do VPS
_, o, _ = ssh.exec_command("date '+%H:%M %Z'")
print("Hora VPS:", o.read().decode().strip())

# Processo
_, o, _ = ssh.exec_command("ps aux | grep retrain_perleague | grep -v grep")
ps = o.read().decode().strip()
print("Processo:", ps if ps else "FINALIZADO")

# Log - ultimas 50 linhas
_, o, _ = ssh.exec_command("tail -50 /tmp/retrain_perleague.log 2>/dev/null")
print("\nLOG:")
print(o.read().decode())

ssh.close()
