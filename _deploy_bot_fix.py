import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

sftp = ssh.open_sftp()
print("📤 Enviando telegram_bot.py...")
sftp.put(r'c:\GitHub\FuteBot\services\telegram_bot.py', '/opt/futebot/services/telegram_bot.py')
sftp.close()
print("✅ Enviado")

print("🔄 Reiniciando...")
_, o, e = ssh.exec_command("systemctl restart futebot")
o.channel.recv_exit_status()

time.sleep(6)

_, o, _ = ssh.exec_command("systemctl is-active futebot")
status = o.read().decode().strip()
print(f"Status: {status}")

_, o, _ = ssh.exec_command("journalctl -u futebot --no-pager -n 10")
print(o.read().decode())

ssh.close()
