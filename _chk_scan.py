import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# Checar se scanner ja foi lancado e ver log
_, o, _ = ssh.exec_command("ls -la /opt/futebot/_run_scanner_now.py 2>/dev/null && echo EXISTE || echo NAO")
print("Script:", o.read().decode().strip())

_, o, _ = ssh.exec_command("ps aux | grep scanner_now | grep -v grep")
ps = o.read().decode().strip()
print("Processo:", ps if ps else "NAO rodando")

_, o, _ = ssh.exec_command("wc -l /tmp/scanner_manual.log 2>/dev/null")
print("Log linhas:", o.read().decode().strip())

_, o, _ = ssh.exec_command("cat /tmp/scanner_manual.log 2>/dev/null")
print("LOG:")
print(o.read().decode())

ssh.close()
