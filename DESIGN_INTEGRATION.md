# GUIA DE INTEGRAÇÃO - Design Moderno no HydroSim-RF

## 🎨 O que foi criado

Um novo módulo `design.py` com componentes modernos que melhoram significativamente a UI/UX:

### Componentes Disponíveis:

1. **`apply_modern_theme()`** - Aplica tema visual profissional ao app
2. **`create_header()`** - Cabeçalho com logos e títulos customizáveis
3. **`create_metric_card()`** - Card individual de métrica com status
4. **`create_metric_row()`** - Linha de múltiplas métricas (responsiva)
5. **`create_section_divider()`** - Divisor visual entre seções
6. **`create_info_box()`** - Caixa de informação/aviso/sucesso/erro
7. **`create_stats_grid()`** - Grade de estatísticas
8. **`create_progress_timeline()`** - Timeline visual de progresso

---

## 🚀 Como Integrar no `hydrosim_rf.py`

### Passo 1: Importar no início do arquivo

```python
from design import apply_modern_theme, create_header, create_metric_row, create_info_box
```

### Passo 2: No início da função `main()`, adicionar logo e tema

```python
def main():
    # Aplicar tema moderno
    apply_modern_theme()
    
    # Criar cabeçalho com logos (se existirem)
    create_header(
        title="HydroSim-RF",
        subtitle="Simulador Híbrido de Inundações",
        logo_main_path="logos/hydrosim_logo.png",  # Ajustar caminho
        logo_secondary_path="logos/hydrolab_logo.png"  # Opcional
    )
    
    # ... resto do código
```

### Passo 3: Exibir métricas em tempo real com cards

**Antes (código atual):**
```python
st.write(f"Tempo decorrido: {current_time}h {current_minute}m")
st.write(f"Área inundada: {flooded_area:.2f} km²")
```

**Depois (com design moderno):**
```python
# Após cada atualização de simulação
metrics = [
    {
        'label': 'Tempo Decorrido',
        'value': f"{current_time}",
        'unit': 'h',
        'status': 'default',
        'icon': '⏱️'
    },
    {
        'label': 'Área Inundada',
        'value': f"{flooded_area:.1f}",
        'unit': 'km²',
        'status': 'warning' if flooded_area > 100 else 'default',
        'icon': '💧'
    },
    {
        'label': 'Volume Total',
        'value': f"{total_volume/1e6:.1f}",
        'unit': 'M m³',
        'status': 'danger' if total_volume > threshold else 'success',
        'icon': '🌊'
    }
]
create_metric_row(metrics)
```

### Passo 4: Adicionar Info Boxes para dicas

```python
if user_uploaded_large_dem:
    create_info_box(
        title="Dica de Performance",
        content="Seu DEM é grande. Considere usar fator de reamostragem 4x ou 8x para acelerar.",
        info_type="warning",
        icon="⚠️"
    )

if simulation_complete:
    create_info_box(
        title="Simulação Concluída!",
        content=f"Foram processadas {total_cycles} ciclos em {elapsed_time} segundos.",
        info_type="success",
        icon="✅"
    )
```

---

## 📁 Estrutura de Logos Esperada

```
hydrosim/
├── logos/
│   ├── hydrosim_logo.png        # Logo principal do projeto
│   ├── hydrosim_logo_alt.png    # Versão alternativa
│   └── hydrolab_logo.png        # Logo da instituição
└── ...
```

### Recomendações para as Logos:

- **Logo Principal**: 800x200px (PNG com fundo transparente)
- **Logo Secundária**: 400x100px (PNG com fundo transparente)
- **Formato**: PNG ou JPEG com alta qualidade
- **Compressão**: Otimizar para web (máx 500KB)

---

## 🎯 Melhoria nos Componentes Visuais

### Cores Adotadas:
- **Primária**: `#1e88e5` (azul profissional)
- **Secundária**: `#26c6da` (ciano moderno)
- **Sucesso**: `#43a047` (verde)
- **Aviso**: `#fbc02d` (amarelo)
- **Erro**: `#e53935` (vermelho)

### Benefícios:
✅ **Temas Responsivos** - Funciona em mobile/tablet/desktop  
✅ **Animações Suaves** - Transições e efeitos modernos  
✅ **Acessibilidade** - Cores com bom contraste  
✅ **Profissionalismo** - Design consistente em todo app  
✅ **Performance** - CSS otimizado, sem dependências extras  

---

## 🔄 Exemplos Completos

### Exemplo 1: Exibir Progresso da Simulação

```python
from design import create_progress_timeline

steps = [
    {'name': 'Preparação', 'status': 'complete'},
    {'name': 'Simulação', 'status': 'active'},
    {'name': 'Pós-processamento', 'status': 'pending'},
    {'name': 'Exportação', 'status': 'pending'},
]
create_progress_timeline(steps)
```

### Exemplo 2: Seção com Título

```python
from design import create_section_divider, create_info_box

create_section_divider("Parâmetros Hidrodinâmicos")
st.slider("Coeficiente de Difusão (α)", 0.01, 1.0, 0.5)

create_section_divider("Configurações de Visualização")
st.selectbox("Basemap", ["Esri", "CartoDB", "OpenStreetMap"])
```

### Exemplo 3: Métricas Lado a Lado

```python
from design import create_stats_grid

stats = {
    'Resolução DEM': '30m',
    'Células da Grade': '1.2M',
    'Tempo de Passo': '60s',
    'Iterações': '2400',
}
create_stats_grid(stats)
```

---

## 🛠️ Customização Avançada

### Alterar Cores Globais

No seu `main()`, antes de usar os componentes, você pode adicionar CSS customizado:

```python
st.markdown("""
<style>
    :root {
        --primary: #your-color !important;
        --secondary: #your-color !important;
    }
</style>
""", unsafe_allow_html=True)
```

### Adicionar Fonts Customizadas

```python
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;500;700&display=swap" rel="stylesheet">
<style>
    * {
        font-family: 'Roboto', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)
```

---

## ✅ Checklist de Integração

- [ ] Copiar logos para `logos/` (opcional)
- [ ] Importar `design.py` no `hydrosim_rf.py`
- [ ] Chamar `apply_modern_theme()` no início de `main()`
- [ ] Chamar `create_header()` com títulos apropriados
- [ ] Substituir `st.write()` por `create_metric_row()` nas métricas
- [ ] Adicionar `create_info_box()` para dicas e avisos
- [ ] Testar responsividade em mobile
- [ ] Ajustar cores se necessário

---

## 🚀 Próximos Passos

1. **Integração Gradual**: Adicionar um componente por vez e testar
2. **Feedback do Usuário**: Coletar opinião sobre cores e layout
3. **Dark Mode** (futuro): Adicionar tema escuro opcional
4. **Exportação**: Melhorar relatórios com design consistente
5. **Documentação**: Atualizar README com screenshots dos novos componentes

---

**Versão**: 1.0.0  
**Data**: Março 2026  
**Autor**: Assistente de IA  
**Mantido por**: Letícia Caldas
