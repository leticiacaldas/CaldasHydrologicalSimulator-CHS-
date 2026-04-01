#!/usr/bin/env bash
set -euo pipefail

echo "🧪 TESTANDO API REST - HydroSim (modo interno)"
echo "==============================================="

/home/leticia/Desktop/hydrosim/venv/bin/python - <<'PY'
import json
import web_server_v3 as ws

app = ws.app

def check_endpoint(client, name, path):
    print(f"\n➡️  {name}: {path}")
    resp = client.get(path)
    print(f"HTTP: {resp.status_code}")
    if resp.status_code < 200 or resp.status_code >= 300:
        print("❌ Endpoint retornou erro")
        print(resp.data[:250])
        raise SystemExit(1)

    data = resp.get_json(silent=True)
    if data is None:
        print("❌ Resposta não é JSON válido")
        print(resp.data[:250])
        raise SystemExit(1)

    if isinstance(data, dict):
        print("✅ JSON válido | chaves:", sorted(list(data.keys()))[:10])
    elif isinstance(data, list):
        print(f"✅ JSON válido | lista com {len(data)} itens")
    else:
        print("✅ JSON válido | tipo:", type(data).__name__)

with app.test_client() as client:
    check_endpoint(client, "Health Check", "/api/health")
    check_endpoint(client, "Listar Simulações", "/api/simulations")

print("\n✅ API REST validada com sucesso!")
PY

echo
if [[ -f "outputs/.database/simulations.db" ]]; then
  echo "✅ Banco encontrado: outputs/.database/simulations.db"
  ls -lh "outputs/.database/simulations.db"
else
  echo "⚠️ Banco não encontrado em outputs/.database/simulations.db"
fi