import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

time.sleep(8)

_, o, _ = ssh.exec_command("journalctl -u futebot --no-pager -n 20")
print(o.read().decode())

ssh.close()
