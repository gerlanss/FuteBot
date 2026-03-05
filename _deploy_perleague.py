"""Deploy trainer.py e predictor.py para o VPS e iniciar treino per-league."""
import paramiko, time, os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("201.76.43.40", username="root", password="Teclado!1")

sftp = ssh.open_sftp()

# 1. Upload dos arquivos modificados
for local, remote in [
    (r"c:\GitHub\FuteBot\models\trainer.py", "/opt/futebot/models/trainer.py"),
    (r"c:\GitHub\FuteBot\models\predictor.py", "/opt/futebot/models/predictor.py"),
]:
    print(f"Uploading {os.path.basename(local)}...")
    sftp.put(local, remote)

sftp.close()
print("Upload concluido")

# 2. Parar servico para liberar recursos
_, o, _ = ssh.exec_command("systemctl stop futebot; echo $?")
print(f"Stop servico: {o.read().decode().strip()}")

# 3. Criar script de treino per-league no VPS
script = '''#!/usr/bin/env python3
"""Treino per-league: 1 modelo por liga + global fallback."""
import sys, time
sys.path.insert(0, "/opt/futebot")

from models.trainer import Trainer
from data.database import Database

start = time.time()
db = Database()
t = Trainer(db)

resultados = t.treinar_por_liga(
    train_seasons=[2020, 2021, 2022, 2023, 2024, 2025],
    test_season=2026
)

elapsed = time.time() - start
print(f"\\nTempo total: {elapsed/60:.1f} min")
print("Concluido!")
'''

sftp = ssh.open_sftp()
with sftp.open('/opt/futebot/_train_perleague.py', 'w') as f:
    f.write(script)
sftp.close()

# 4. Executar em background
cmd = "cd /opt/futebot && nohup ./venv/bin/python -u _train_perleague.py > /tmp/train_perleague.log 2>&1 &"
_, o, _ = ssh.exec_command(cmd)
o.read()
time.sleep(3)

# 5. Verificar se iniciou
_, o, _ = ssh.exec_command("ps aux | grep train_perleague | grep -v grep")
ps = o.read().decode()
if "train_perleague" in ps:
    print("Treino per-league INICIADO!")
    print("Log: /tmp/train_perleague.log")
else:
    print("ERRO - verificando log:")
    _, o, _ = ssh.exec_command("head -30 /tmp/train_perleague.log")
    print(o.read().decode())

ssh.close()
