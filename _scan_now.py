import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# Disparar scanner em background
_, o, _ = ssh.exec_command("cd /opt/futebot && nohup ./venv/bin/python -u _run_scanner_now.py > /tmp/scanner_manual.log 2>&1 &")
o.read()
time.sleep(5)

# Checar processo
_, o, _ = ssh.exec_command("ps aux | grep scanner_now | grep -v grep")
ps = o.read().decode().strip()
print("PS:", ps if ps else "JA TERMINOU")

# Checar log
_, o, _ = ssh.exec_command("cat /tmp/scanner_manual.log 2>/dev/null")
print("LOG:", o.read().decode())

ssh.close()
