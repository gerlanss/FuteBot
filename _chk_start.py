import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# Ultimas 40 linhas — deve ter o momento do /start das 15:22
_, o, _ = ssh.exec_command("journalctl -u futebot --no-pager -n 40")
print(o.read().decode())

ssh.close()
