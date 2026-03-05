"""Deploy bot.py e bulk_download.py corrigidos no VPS e reinicia o serviço."""
import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

sftp = ssh.open_sftp()

# Enviar arquivos corrigidos
print("📤 Enviando bot.py...")
sftp.put(r'c:\GitHub\FuteBot\bot.py', '/opt/futebot/bot.py')
print("📤 Enviando bulk_download.py...")
sftp.put(r'c:\GitHub\FuteBot\data\bulk_download.py', '/opt/futebot/data/bulk_download.py')
sftp.close()
print("✅ Arquivos enviados")

# Reiniciar serviço
print("\n🔄 Reiniciando futebot...")
_, o, e = ssh.exec_command("systemctl restart futebot")
o.channel.recv_exit_status()
err = e.read().decode().strip()
if err:
    print(f"⚠️ stderr: {err}")

time.sleep(5)

# Verificar status
_, o, _ = ssh.exec_command("systemctl is-active futebot")
status = o.read().decode().strip()
print(f"Status: {status}")

# Ultimas 15 linhas do log
_, o, _ = ssh.exec_command("journalctl -u futebot --no-pager -n 15")
print("\n=== LOG ===")
print(o.read().decode())

ssh.close()
