import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

_, o, _ = ssh.exec_command("date '+%H:%M'")
print("HORA VPS:", o.read().decode().strip())

_, o, _ = ssh.exec_command("ps aux | grep train_perleague | grep -v grep")
ps = o.read().decode().strip()
print("PROCESSO:", ps if ps else "FINALIZADO")

_, o, _ = ssh.exec_command("tail -60 /tmp/train_perleague.log 2>/dev/null")
print("\nLOG:")
print(o.read().decode())

ssh.close()
