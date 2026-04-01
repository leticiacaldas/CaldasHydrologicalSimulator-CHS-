#!/usr/bin/env python3
"""
Script para re-gerar visualizações com as correções aplicadas.
Usa os dados já salvos em outputs/test_run/results.npz
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent))

# Carregar dados salvos
npz_path = "outputs/test_run/results.npz"
if not Path(npz_path).exists():
    print(f"❌ Arquivo {npz_path} não encontrado!")
    sys.exit(1)

data = np.load(npz_path, allow_pickle=True)
print("✅ Arquivos carregados:")
print(f"   - DEM shape: {data['dem'].shape}")
print(f"   - Water final shape: {data['water_final'].shape}")
print(f"   - Sources shape: {data['sources'].shape}")

dem = data['dem']
water_height = data['water_final']
sources = data['sources']
prob = data.get('probability', np.ones_like(dem) * 0.5)  # Fallback

print("\n📊 Estatísticas dos dados:")
print(f"   - DEM: min={dem.min():.2f}, max={dem.max():.2f}, média={np.nanmean(dem):.2f}")
print(f"   - Water: min={water_height.min():.4f}, max={water_height.max():.4f}, média={np.nanmean(water_height):.4f}")
print(f"   - Prob: min={prob.min():.4f}, max={prob.max():.4f}, média={np.nanmean(prob):.4f}")

# ==================== GERAR VISUALIZAÇÕES ====================

print("\n🎨 Gerando visualizações...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ========== PAINEL 1: DEM ==========
dem_float = dem.astype(np.float32)
vmin_dem, vmax_dem = np.nanpercentile(dem_float, (5, 95)) if np.isfinite(dem_float).any() else (0, 1)
axes[0, 0].imshow(dem_float, cmap='terrain', vmin=vmin_dem, vmax=vmax_dem)
axes[0, 0].set_title('DEM (Digital Elevation Model)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('X (cells)')
axes[0, 0].set_ylabel('Y (cells)')

# ========== PAINEL 2: Fontes de Chuva ==========
axes[0, 1].imshow(dem_float, cmap='gray', alpha=0.5, vmin=vmin_dem, vmax=vmax_dem)
axes[0, 1].contourf(sources.astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.7)
axes[0, 1].set_title('Rainfall Sources', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('X (cells)')
axes[0, 1].set_ylabel('Y (cells)')

# ========== PAINEL 3: Profundidade de Água ==========
water_threshold = 0.001  # 0.1cm
water_display = water_height.copy()

# Colormap customizado para água
water_cmap = mcolors.LinearSegmentedColormap.from_list(
    "water_layer",
    [(1.0, 1.0, 1.0), (0.78, 0.98, 1.00), (0.00, 0.92, 0.90), (0.00, 0.62, 1.00),
     (0.00, 0.30, 0.72)],
    N=256,
)
water_cmap.set_under("white")  # Valores baixos = branco
water_cmap.set_bad("white")     # NaN = branco

# Mostrar DEM como background
axes[1, 0].imshow(dem_float, cmap='terrain', vmin=vmin_dem, vmax=vmax_dem, alpha=0.85, zorder=1)

# Sobrepor água (SEM máscara - mostrar dados reais)
water_display[water_display < water_threshold] = 0
water_max = np.nanmax(water_display) if water_display.max() > water_threshold else 0.2
im1 = axes[1, 0].imshow(water_display, cmap=water_cmap, vmin=0, vmax=water_max, alpha=0.8, zorder=2)
cbar1 = plt.colorbar(im1, ax=axes[1, 0], label='Water Depth (m)', fraction=0.046, pad=0.04)
axes[1, 0].set_title('Final Water Depth', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('X (cells)')
axes[1, 0].set_ylabel('Y (cells)')

# ========== PAINEL 4: Probabilidade de Inundação ==========
# NÃO mascarar - mostrar todos os dados
prob_cmap = cm.get_cmap('RdYlGn_r').copy()
prob_cmap.set_under("white")   # Valores abaixo de vmin aparecem branco
prob_cmap.set_bad("gray")      # NaN values aparecem cinza

# Mostrar DEM como background
axes[1, 1].imshow(dem_float, cmap='gray', vmin=vmin_dem, vmax=vmax_dem, alpha=0.3, zorder=1)

# Sobrepor probabilidade (SEM máscara - mostrar todos os pixels)
prob_threshold = 0.01  # Apenas mostrar onde prob > 1%
prob_display = prob.copy()
prob_display[prob_display < prob_threshold] = 0  # Fundo branco
im2 = axes[1, 1].imshow(prob_display, cmap=prob_cmap, vmin=0, vmax=1, zorder=2)
cbar2 = plt.colorbar(im2, ax=axes[1, 1], label='Probability', fraction=0.046, pad=0.04)
axes[1, 1].set_title('Flood Probability', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('X (cells)')
axes[1, 1].set_ylabel('Y (cells)')

plt.tight_layout()

# Salvar
output_path = "outputs/test_run/results_visualization_fixed.png"
Path("outputs/test_run").mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Salvo: {output_path}")

plt.close()

print("\n✨ Visualizações geradas com sucesso!")
print("\n📊 Resumo dos painéis:")
print("   1️⃣  DEM: Terrain colormap com altitude")
print("   2️⃣  Fontes: DEM cinza + fontes vermelhas")
print("   3️⃣  Água: DEM terrain + água azul (claro→escuro)")
print("   4️⃣  Probabilidade: DEM cinza + probabilidade RdYlGn_r")
