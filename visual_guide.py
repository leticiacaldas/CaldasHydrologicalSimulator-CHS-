#!/usr/bin/env python3
"""
visual_guide.py - Guia Visual do Design System

Mostra visualmente como integrar os componentes
"""

HEADER_EXAMPLE = """
╔════════════════════════════════════════════════════════════════════════╗
║                      HydroSim-RF Design System                         ║
║     Simulador Híbrido de Inundações com Design Moderno                 ║
╚════════════════════════════════════════════════════════════════════════╝
"""

METRIC_CARD_EXAMPLE = """
┌──────────────────────────────────────────┐
│ 💧 Área Inundada                         │
│ 125.5 km² [STATUS: WARNING]              │
└──────────────────────────────────────────┘
"""

METRIC_ROW_EXAMPLE = """
┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ ⏱️  Tempo       │ │ 💧 Área        │ │ 🌊 Volume      │ │ 🤖 RF Score    │
│ 2h 45m         │ │ 125.5 km²      │ │ 2.3M m³        │ │ 94.2%          │
│ [DEFAULT]      │ │ [WARNING]      │ │ [DANGER]       │ │ [SUCCESS]      │
└────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘
"""

INFO_BOX_EXAMPLE = """
┌──────────────────────────────────────────────────────────────────┐
│ ℹ️  Informação                                                   │
├──────────────────────────────────────────────────────────────────┤
│ Este é um exemplo de info box para dicas e contexto geral.       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ ✅ Sucesso!                                                      │
├──────────────────────────────────────────────────────────────────┤
│ A simulação foi executada com sucesso em 2h 45 minutos!          │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ ⚠️  Atenção                                                      │
├──────────────────────────────────────────────────────────────────┤
│ Seu DEM é grande. Use fator de reamostragem 8x para acelerar.    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ ❌ Erro                                                          │
├──────────────────────────────────────────────────────────────────┤
│ Falha ao carregar arquivo. Verifique o formato GeoTIFF.          │
└──────────────────────────────────────────────────────────────────┘
"""

TIMELINE_EXAMPLE = """
TIMELINE DE SIMULAÇÃO:

    ✓         ✓         ◐         ○
    ●─────────●─────────●─────────●
    │         │         │         │
Preparação  Simulação  Pós-Proc  Exporta
(100%)      (100%)     (75%)     (0%)

LEGENDA:
✓ = Completo (verde)
◐ = Em progresso (azul)
○ = Pendente (cinza)
"""

STATS_GRID_EXAMPLE = """
ESTATÍSTICAS DA SIMULAÇÃO

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Resolução DEM    │  │ Células da Grade │  │ Tempo de Passo   │
│ 30m              │  │ 1.2M             │  │ 60s              │
└──────────────────┘  └──────────────────┘  └──────────────────┘

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Iterações        │  │ Memória Usada    │  │ CPU Médio        │
│ 2,400            │  │ 850 MB           │  │ 65%              │
└──────────────────┘  └──────────────────┘  └──────────────────┘
"""

INTEGRATION_FLOW = """
FLUXO DE INTEGRAÇÃO
═══════════════════════════════════════════════════════════════════

  1️⃣  ANTES                          2️⃣  DEPOIS

hydrosim_rf.py                   hydrosim_rf.py
│                                │
├─ import streamlit               ├─ import streamlit
├─ run simulation                 ├─ import design ✨
│                                 ├─ run simulation
├─ st.write("Área: 125 km²")     │
├─ st.write("Volume: 2.3M m³")   ├─ apply_modern_theme() ✨
├─ st.write("Tempo: 2h 45m")     ├─ create_header() ✨
│                                 ├─ create_metric_row(metrics) ✨
└─ Results shown as text          │
                                  ├─ create_info_box() ✨
                                  │
                                  └─ Results shown with design ✨

BENEFÍCIO: Informações críticas em destaque, aspecto profissional
"""

COLOR_PALETTE = """
PALETA DE CORES DO DESIGN SYSTEM
═════════════════════════════════════════════════════════════════

PRIMARY (Ação Principal)
████ #1e88e5 (Azul)
Botões, links, ação primária

SECONDARY (Secundário)  
████ #26c6da (Ciano)
Destaque, gradientes

SUCCESS (Sucesso)
████ #43a047 (Verde)
Operações bem-sucedidas

WARNING (Aviso)
████ #fbc02d (Amarelo)
Atenção necessária

DANGER (Erro)
████ #e53935 (Vermelho)
Problemas críticos

DARK TEXT (Texto)
████ #1a1a1a (Preto)
Texto principal

LIGHT BG (Fundo)
████ #f5f7fa (Cinza claro)
Fundo da página

BORDER (Bordas)
████ #e0e0e0 (Cinza)
Separadores
"""

RESPONSIVE_DESIGN = """
RESPONSIVIDADE DO DESIGN
════════════════════════════════════════════════════════════════════

DESKTOP (> 1024px)                TABLET (768-1024px)          MOBILE (< 768px)
┌─────────────────────────────┐   ┌──────────────┐            ┌─────────┐
│ [LOGO] TÍTULO [LOGO]        │   │ TÍTULO       │            │ TÍTULO  │
├─────────────────────────────┤   ├──────────────┤            ├─────────┤
│ ┌─────┐ ┌─────┐ ┌─────┐     │   │ ┌─────┐     │            │ ┌─────┐ │
│ │ KPI │ │ KPI │ │ KPI │     │   │ │ KPI │     │            │ │ KPI │ │
│ └─────┘ └─────┘ └─────┘     │   │ └─────┘     │            │ └─────┘ │
│ ┌─────┐ ┌─────┐ ┌─────┐     │   │ ┌─────┐     │            │ ┌─────┐ │
│ │ KPI │ │ KPI │ │ KPI │     │   │ │ KPI │     │            │ │ KPI │ │
│ └─────┘ └─────┘ └─────┘     │   │ └─────┘     │            │ └─────┘ │
├─────────────────────────────┤   ├──────────────┤            ├─────────┤
│ Conteúdo principal          │   │ Conteúdo     │            │ Conteúdo│
│ (2-3 colunas)               │   │ (1 coluna)   │            │ (1 col) │
└─────────────────────────────┘   └──────────────┘            └─────────┘

✅ Adapta automaticamente ao tamanho da tela
✅ Sem scroll horizontal
✅ Touch-friendly em mobile
"""

COMPONENT_MATRIX = """
MATRIZ DE COMPONENTES
════════════════════════════════════════════════════════════════════

COMPONENTE          │ USAR PARA              │ EXEMPLO
────────────────────┼────────────────────────┼──────────────────
apply_modern_theme  │ Setup inicial          │ main() início
create_header       │ Topo da página         │ Logo + título
create_metric_card  │ Uma métrica            │ "Área: 125 km²"
create_metric_row   │ Múltiplas métricas     │ 3-4 KPIs
create_section_div  │ Separar seções         │ Entre abas
create_info_box     │ Dicas/alertas          │ "DEM grande"
create_stats_grid   │ Configurações          │ Resolução, tempo
create_progress_tm  │ Fluxo de passos        │ Prep→Sim→Export
────────────────────┴────────────────────────┴──────────────────

✅ Use quantos precisar
✅ Combinar para layouts complexos
✅ Reutilizar em múltiplas páginas
"""

QUICK_START_CODE = """
QUICK START - 3 LINHAS DE CÓDIGO
════════════════════════════════════════════════════════════════════

1. Importe:
   from design import apply_modern_theme, create_header, create_metric_row

2. Configure tema:
   apply_modern_theme()
   create_header("HydroSim-RF", "Simulador de Inundações")

3. Exiba métricas:
   metrics = [{'label': 'Área', 'value': '125.5', 'unit': 'km²', 'icon': '💧'}]
   create_metric_row(metrics)

✅ Pronto! Seu app tem design moderno.
"""

def print_visual_guide():
    """Imprime o guia visual completo"""
    
    print("\n" + "="*80)
    print("        🎨 HYDROSIM-RF - VISUAL DESIGN GUIDE")
    print("="*80 + "\n")
    
    print(HEADER_EXAMPLE)
    print(QUICK_START_CODE)
    print(METRIC_CARD_EXAMPLE)
    print(METRIC_ROW_EXAMPLE)
    print(INFO_BOX_EXAMPLE)
    print(TIMELINE_EXAMPLE)
    print(STATS_GRID_EXAMPLE)
    print(INTEGRATION_FLOW)
    print(COLOR_PALETTE)
    print(RESPONSIVE_DESIGN)
    print(COMPONENT_MATRIX)
    
    print("\n" + "="*80)
    print("        📚 Para mais informações:")
    print("        - DESIGN_SUMMARY.md")
    print("        - DESIGN_INTEGRATION.md")
    print("        - example_design_showcase.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_visual_guide()
    
    print("💡 Próximas ações:")
    print("  1. Leia DESIGN_SUMMARY.md")
    print("  2. Execute: streamlit run example_design_showcase.py")
    print("  3. Integre no hydrosim_rf.py seguindo DESIGN_INTEGRATION.md")
    print()
