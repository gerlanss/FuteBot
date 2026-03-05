"""Deploy config.py e scheduler.py atualizados pro VPS."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

sftp = ssh.open_sftp()

# 1. config.py
with open(r'c:\GitHub\FuteBot\config.py', 'r', encoding='utf-8') as f:
    content = f.read()
with sftp.open('/opt/futebot/config.py', 'w') as f:
    f.write(content)
print("config.py deployado")

# 2. pipeline/scheduler.py
with open(r'c:\GitHub\FuteBot\pipeline\scheduler.py', 'r', encoding='utf-8') as f:
    content = f.read()
with sftp.open('/opt/futebot/pipeline/scheduler.py', 'w') as f:
    f.write(content)
print("scheduler.py deployado")

sftp.close()

# Verificar se treino per-league ainda roda antes de reiniciar servico
_, o, _ = ssh.exec_command("ps aux | grep train_perleague | grep -v grep")
ps = o.read().decode().strip()
if ps:
    print("\nTreino per-league AINDA RODANDO - servico sera reiniciado quando terminar")
    print(ps)
else:
    # Reiniciar servico
    _, o, _ = ssh.exec_command("systemctl restart futebot; sleep 2; systemctl is-active futebot")
    print(f"\nServico futebot: {o.read().decode().strip()}")

# Confirmar RETREINO_HORA no config deployado
_, o, _ = ssh.exec_command("grep RETREINO_HORA /opt/futebot/config.py")
print(f"\nConfig VPS: {o.read().decode().strip()}")

_, o, _ = ssh.exec_command("grep 'retreino_semanal' /opt/futebot/pipeline/scheduler.py")
print(f"Scheduler VPS: {o.read().decode().strip()}")

ssh.close()
