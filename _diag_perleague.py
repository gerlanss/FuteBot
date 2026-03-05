import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# 1. Verificar se trainer.py foi deployado corretamente
_, o, _ = ssh.exec_command("grep 'treinar_por_liga' /opt/futebot/models/trainer.py | head -5")
print("1. treinar_por_liga no trainer.py:")
print(o.read().decode() or "  NAO ENCONTRADO!")

# 2. Verificar se predictor.py tem per-league
_, o, _ = ssh.exec_command("grep 'league_id' /opt/futebot/models/predictor.py | head -5")
print("2. league_id no predictor.py:")
print(o.read().decode() or "  NAO ENCONTRADO!")

# 3. Verificar se o script de treino existe
_, o, _ = ssh.exec_command("ls -la /opt/futebot/_retrain_perleague.py 2>/dev/null")
print("3. Script de treino:")
print(o.read().decode() or "  NAO ENCONTRADO!")

# 4. Tentar importar trainer
_, o, e = ssh.exec_command("cd /opt/futebot && ./venv/bin/python -c 'from models.trainer import Trainer; print(\"Import OK\"); print(hasattr(Trainer, \"treinar_por_liga\"))'")
out = o.read().decode()
err = e.read().decode()
print("4. Import teste:")
print(out if out else "  (sem output)")
if err:
    # Filtrar UserWarnings do xgboost
    lines = [l for l in err.split('\n') if 'UserWarning' not in l and l.strip()]
    if lines:
        print("  ERRO:", '\n'.join(lines[:10]))

# 5. Verificar log nohup
_, o, _ = ssh.exec_command("cat /tmp/nohup_perleague.out 2>/dev/null | tail -20")
alt_log = o.read().decode()
if alt_log.strip():
    print("5. nohup output:")
    print(alt_log)

# 6. Checar se os scripts temporários foram deletados (limpamos antes!)
_, o, _ = ssh.exec_command("ls /opt/futebot/_retrain*.py 2>/dev/null")
print("6. Scripts _retrain no VPS:")
print(o.read().decode() or "  NENHUM (foram limpos!)")

ssh.close()
