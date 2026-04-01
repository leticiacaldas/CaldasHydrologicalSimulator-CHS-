#!/usr/bin/env python3
"""
Script para testar exportação RGBA GeoTIFF da lâmina de água.
Demonstra como usar a função _water_rgba_geotiff_bytes() para gerar lâmina estilizada.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Importar a função
from web_server_v3 import _water_rgba_geotiff_bytes

# Carregar dados salvos
npz_path = "outputs/test_run/results.npz"
if not Path(npz_path).exists():
    print(f"❌ Arquivo {npz_path} não encontrado!")
    sys.exit(1)

data = np.load(npz_path, allow_pickle=True)
water_height = data['water_final']

print("✅ Dados carregados")
print(f"   - Water: min={water_height.min():.4f}, max={water_height.max():.4f}, média={np.nanmean(water_height):.4f}")

# Gerar RGBA bytes
try:
    vmin = 0.001  # 0.1cm
    vmax = float(np.nanmax(water_height)) if np.nanmax(water_height) > vmin else 0.2
    
    rgba_bytes = _water_rgba_geotiff_bytes(water_height, vmin=vmin, vmax=vmax)
    
    # Salvar para teste
    output_path = Path("outputs/test_run/lamina_agua_rgba_test.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(rgba_bytes)
    
    print(f"✅ RGBA GeoTIFF gerado com sucesso!")
    print(f"   - Tamanho: {len(rgba_bytes) / 1024:.1f} KB")
    print(f"   - Salvo em: {output_path}")
    print(f"   - vmin: {vmin} m")
    print(f"   - vmax: {vmax} m")
    
except Exception as e:
    print(f"❌ Erro ao gerar RGBA: {e}")
    import traceback
    traceback.print_exc()
