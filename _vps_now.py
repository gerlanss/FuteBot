import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

_, o, _ = ssh.exec_command("date '+%H:%M'")
print("HORA:", o.read().decode().strip())

_, o, _ = ssh.exec_command("ps aux | grep python | grep -v grep")
print("PROCESSOS PYTHON:")
print(o.read().decode())

_, o, _ = ssh.exec_command("wc -l /tmp/retrain_perleague.log 2>/dev/null")
print("LINHAS LOG:", o.read().decode().strip())

_, o, _ = ssh.exec_command("tail -20 /tmp/retrain_perleague.log 2>/dev/null")
log = o.read().decode()
print("LOG TAIL:", log if log.strip() else "(vazio)")

_, o, _ = ssh.exec_command("ls /opt/futebot/_retrain_perleague.py 2>/dev/null && echo EXISTE || echo NAO_EXISTE")
print("SCRIPT:", o.read().decode().strip())

ssh.close()
