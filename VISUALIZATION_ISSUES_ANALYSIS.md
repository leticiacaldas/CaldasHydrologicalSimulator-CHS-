# 🔴 ANÁLISE CRÍTICA: Problemas de Visualização

## ❌ PROBLEMAS IDENTIFICADOS

### 1. **FALTA DE `.set_under()` NO COLORMAP (CRÍTICO)**

**Função**: `_generate_visualizations()` - linha 978

```python
# ❌ ATUAL (ERRADO):
im = axes[1, 1].imshow(prob_masked, cmap='RdYlGn_r', vmin=0, vmax=1)

# ✅ CORRETO:
prob_cmap = plt.get_cmap('RdYlGn_r').copy()
prob_cmap.set_under((0, 0, 0, 0.0))  # Transparência para valores mascarados
im = axes[1, 1].imshow(prob_masked, cmap=prob_cmap, vmin=0, vmax=1)
```

**Por que quebra**: Sem `.set_under()`, matplotlib usa a cor "mínima" do colormap para pixels mascarados. Como o colormap vai de vermelho (0) até verde (1), os pixels mascarados ficam **vermelhos** ou com outra cor default.

---

### 2. **COLORMAP ERRADO PARA ÁGUA NA ANIMAÇÃO**

**Função**: `_generate_animation_improved()` - linha 1045

```python
# ❌ ATUAL (ERRADO):
cmap = mpl_cm.get_cmap('plasma')  # Sem .set_under()!
im = ax.imshow(water_mask, cmap=cmap, vmin=threshold, vmax=max_water_depth, alpha=0.85, zorder=2)

# ✅ CORRETO (como no código antigo):
water_cmap = mcolors.LinearSegmentedColormap.from_list(
    "water_layer",
    [
        (0.78, 0.98, 1.00),  # Ciano claro
        (0.00, 0.92, 0.90),  # Turquesa
        (0.00, 0.62, 1.00),  # Azul médio
        (0.00, 0.30, 0.72),  # Azul profundo
    ],
    N=256,
)
water_cmap.set_under((0, 0, 0, 0.0))  # ← ESSENCIAL!
water_cmap.set_bad((0, 0, 0, 0.0))     # ← Para NaN values

im = ax.imshow(water_mask, cmap=water_cmap, vmin=threshold, vmax=max_water_depth, alpha=0.85, zorder=2)
```

**Por que quebra**: 
- Plasma é ótimo para visualização mas **não respeita mascaras bem**
- Sem `.set_under()`, todos os pixels <= threshold ficam com a cor mínima (roxo escuro em plasma)
- Resultado: GIF todo roxo em vez de mostrar apenas água

---

### 3. **PROBLEMA NA VISUALIZAÇÃO DE ÁGUA (web_server_v3.py linha 976)**

```python
# ❌ ATUAL (ERRADO):
if ortho_rgb is not None:
    axes[1, 0].imshow(ortho_rgb)
    axes[1, 0].imshow(model.water_height, cmap='Blues', alpha=0.75)  # Sem mascara!
else:
    axes[1, 0].imshow(model.water_height, cmap='Blues')

# ✅ CORRETO:
if ortho_rgb is not None:
    axes[1, 0].imshow(ortho_rgb)
    # Mascarar água para mostrar apenas onde acumula
    water_masked = np.ma.masked_where(model.water_height <= 0.01, model.water_height)
    water_cmap_blue = plt.get_cmap('Blues').copy()
    water_cmap_blue.set_under((0, 0, 0, 0.0))
    axes[1, 0].imshow(water_masked, cmap=water_cmap_blue, alpha=0.5)
else:
    water_masked = np.ma.masked_where(model.water_height <= 0.01, model.water_height)
    water_cmap_blue = plt.get_cmap('Blues').copy()
    water_cmap_blue.set_under((0, 0, 0, 0.0))
    axes[1, 0].imshow(water_masked, cmap=water_cmap_blue)
axes[1, 0].set_title('Final Water Depth')
```

---

### 4. **FALTA NORMALIZAÇÃO POWERNOM PARA CONTRASTE**

**Função**: `_generate_animation_improved()` - linha 1045

```python
# ❌ ATUAL (ERRADO - sem PowerNorm):
im = ax.imshow(water_mask, cmap=cmap, vmin=threshold, vmax=max_water_depth, alpha=0.85)

# ✅ CORRETO:
try:
    from matplotlib.colors import PowerNorm
    norm = PowerNorm(gamma=0.7, vmin=threshold, vmax=max_water_depth)
    im = ax.imshow(water_mask, cmap=water_cmap, norm=norm, alpha=0.85, zorder=2)
except Exception:
    im = ax.imshow(water_mask, cmap=water_cmap, vmin=threshold, vmax=max_water_depth, alpha=0.85, zorder=2)
```

**Por que muda**: PowerNorm com gamma < 1 realça diferenças em valores baixos (água rasa fica mais visível).

---

## 📊 RESUMO DAS CORREÇÕES NECESSÁRIAS

| Função | Linha | Problema | Solução |
|--------|-------|----------|---------|
| `_generate_visualizations()` | 978 | Falta `.set_under()` em cmap | Adicionar `.copy()` e `.set_under()` |
| `_generate_visualizations()` | 976-977 | Água sem máscara | Mascarar e usar `.set_under()` |
| `_generate_animation_improved()` | 1045 | Plasma sem `.set_under()` | Usar colormap customizado + `.set_under()` |
| `_generate_animation_improved()` | 1045 | Sem PowerNorm | Adicionar PowerNorm(gamma=0.7) |

---

## 🎯 RESULTADO ESPERADO APÓS CORREÇÕES

✅ **Antes** (QUEBRADO):
- Água toda amarela na visualização estática
- GIF toda roxo (plasma colormap sem mascara)
- Probabilidade toda vermelha fora da área válida

✅ **Depois** (CORRETO):
- Água azul/turquesa apenas onde acumula
- GIF mostra transição gradual de azul claro → escuro conforme profundidade
- Probabilidade com cores APENAS dentro da área válida do DEM
- Áreas inválidas transparentes/não mostradas

