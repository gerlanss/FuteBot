"""Rodar scanner manualmente para gerar tips de hoje."""
import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('201.76.43.40', username='root', password='Teclado!1')

# Rodar scanner via o bot.py (trigger manual)
script = '''#!/usr/bin/env python3
import sys
sys.path.insert(0, "/opt/futebot")

from pipeline.scanner import Scanner
from data.database import Database

db = Database()
scanner = Scanner(db)

print("Executando scanner...")
resultado = scanner.executar(dias_adiante=0)

tips = resultado.get("tips", [])
print(f"Tips encontradas: {len(tips)}")

if tips:
    msgs = scanner.formatar_relatorio(resultado)
    print(f"Mensagens formatadas: {len(msgs)}")
    
    # Enviar via Telegram
    from services.telegram_bot import enviar_mensagem_direta
    import asyncio
    
    async def enviar():
        for msg in msgs:
            await enviar_mensagem_direta(msg)
            print(f"  Enviada: {msg[:80]}...")
    
    asyncio.run(enviar())
    print("Tips enviadas!")
else:
    print("Nenhuma tip encontrada para hoje")
    print(f"Jogos analisados: {resultado.get('jogos_analisados', 0)}")
    print(f"Jogos com odds: {resultado.get('jogos_com_odds', 0)}")
    print(f"Motivo: {resultado.get('motivo', 'desconhecido')}")
'''

sftp = ssh.open_sftp()
with sftp.open('/opt/futebot/_run_scanner_now.py', 'w') as f:
    f.write(script)
sftp.close()

_, o, e = ssh.exec_command("cd /opt/futebot && ./venv/bin/python -u _run_scanner_now.py 2>&1", timeout=120)
out = o.read().decode()
err = e.read().decode()

print(out)
if err:
    lines = [l for l in err.split('\n') if 'Warning' not in l and 'UserWarning' not in l and l.strip()]
    if lines:
        print("ERROS:", '\n'.join(lines[:20]))

ssh.close()
