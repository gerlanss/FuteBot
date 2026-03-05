import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

sftp = ssh.open_sftp()
print("📤 Enviando scanner.py...")
sftp.put(r'c:\GitHub\FuteBot\pipeline\scanner.py', '/opt/futebot/pipeline/scanner.py')
sftp.close()
print("✅ Enviado")

print("🔄 Reiniciando...")
_, o, e = ssh.exec_command("systemctl restart futebot")
o.channel.recv_exit_status()
time.sleep(5)

_, o, _ = ssh.exec_command("systemctl is-active futebot")
print(f"Status: {o.read().decode().strip()}")

ssh.close()
