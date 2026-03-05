"""Dispara treino per-league no VPS."""
import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# Parar servico para liberar recursos
_, o, _ = ssh.exec_command("systemctl stop futebot; echo $?")
print("Stop futebot:", o.read().decode().strip())

# Criar script no VPS
script = '''#!/usr/bin/env python3
"""Treino per-league — roda no VPS."""
import sys, time
sys.path.insert(0, "/opt/futebot")

from models.trainer import Trainer
from data.database import Database

start = time.time()
db = Database()
trainer = Trainer(db)

print("=" * 60)
print("TREINO PER-LEAGUE")
print("Seasons treino: 2020-2025")
print("Season teste: 2026")
print("=" * 60)
print(flush=True)

resultados = trainer.treinar_por_liga(
    train_seasons=[2020, 2021, 2022, 2023, 2024, 2025],
    test_season=2026
)

elapsed = time.time() - start
print()
print("=" * 60)
print(f"CONCLUIDO em {elapsed/60:.1f} min")
print("=" * 60)

global_r = resultados.get("global", {})
print(f"\\nModelo GLOBAL:")
print(f"  Aprovados: {global_r.get('modelos_aprovados', [])}")
print(f"  Reprovados: {global_r.get('modelos_reprovados', [])}")

for lid, r in sorted(resultados.items()):
    if lid == "global":
        continue
    aprov = r.get("modelos_aprovados", [])
    reprov = r.get("modelos_reprovados", [])
    jogos = r.get("jogos_treino", "?")
    print(f"\\nLiga {lid} ({jogos} jogos treino):")
    print(f"  Aprovados ({len(aprov)}): {aprov}")
    if reprov:
        print(f"  Reprovados ({len(reprov)}): {reprov}")

print("\\nFinalizado!")
'''

sftp = ssh.open_sftp()
with sftp.open('/opt/futebot/_retrain_perleague.py', 'w') as f:
    f.write(script)
sftp.close()
print("Script enviado")

# Rodar em background
cmd = "cd /opt/futebot && nohup ./venv/bin/python -u _retrain_perleague.py > /tmp/retrain_perleague.log 2>&1 &"
_, o, _ = ssh.exec_command(cmd)
o.read()
time.sleep(3)

# Verificar
_, o, _ = ssh.exec_command("ps aux | grep retrain_perleague | grep -v grep")
ps = o.read().decode().strip()
if "retrain_perleague" in ps:
    print("Treino per-league INICIADO!")
    _, o, _ = ssh.exec_command("head -10 /tmp/retrain_perleague.log")
    print(o.read().decode())
else:
    print("ERRO - verificando log:")
    _, o, _ = ssh.exec_command("cat /tmp/retrain_perleague.log 2>&1")
    print(o.read().decode())

ssh.close()
