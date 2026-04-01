#!/usr/bin/env python3
"""
Script para gerar dados de teste realistas com água acumulada em vales
e probabilidade de inundação variável.
"""

import numpy as np
from pathlib import Path

# Criar DEM sintético com vales
dem = np.zeros((100, 100), dtype=np.float32)
for i in range(100):
    for j in range(100):
        # Vale diagonal
        dem[i, j] = 80 + 30 * np.sin((i - 50) / 25) + 20 * np.cos((j - 50) / 30)

# Água acumulada: mais profunda nos vales
water = np.zeros((100, 100), dtype=np.float32)
for i in range(100):
    for j in range(100):
        # Mais água onde o DEM é mais baixo
        dem_norm = (dem[i, j] - dem.min()) / (dem.max() - dem.min())
        water[i, j] = max(0, 0.15 * (1 - dem_norm) + 0.01 * np.random.random())

# Probabilidade de inundação correlacionada com água
prob = np.zeros((100, 100), dtype=np.float32)
for i in range(100):
    for j in range(100):
        # Probabilidade aumenta com água acumulada
        prob[i, j] = min(1.0, water[i, j] * 2 + 0.1 * np.random.random())

# Fontes de chuva (random)
sources = np.random.random((100, 100)) > 0.9

# Dados de história (para animação)
history = []
for t in range(10):
    water_snapshot = water * (t + 1) / 10  # Aumenta gradualmente
    history.append({
        "time_minutes": t * 10,
        "water_height_snapshot": water_snapshot
    })

# Salvar
output_path = Path("outputs/test_run/results.npz")
output_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    output_path,
    dem=dem,
    water_final=water,
    probability=prob,
    sources=sources,
)

print(f"✅ Dados sintéticos gerados em {output_path}")
print(f"\nEstatísticas:")
print(f"DEM: min={dem.min():.2f}, max={dem.max():.2f}, média={dem.mean():.2f}")
print(f"Water: min={water.min():.4f}, max={water.max():.4f}, média={water.mean():.4f}")
print(f"Prob: min={prob.min():.4f}, max={prob.max():.4f}, média={prob.mean():.4f}")
print(f"Water > 0.01: {(water > 0.01).sum()} pixels")
print(f"Prob > 0.1: {(prob > 0.1).sum()} pixels")
