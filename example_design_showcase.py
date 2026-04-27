"""
example_design_showcase.py - Demonstration dos Componentes de Design

Execute com: streamlit run example_design_showcase.py

Este arquivo demonstra todos os components visuais disponíveis no design.py
"""

import streamlit as st
from design import (
    apply_modern_theme,
    create_header,
    create_metric_card,
    create_metric_row,
    create_section_divider,
    create_info_box,
    create_stats_grid,
    create_progress_timeline,
)


def main():
    """Main page com demonstração dos components."""
    
    # Configure page
    st.set_page_config(
        page_title="HydroSim-RF Design System",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Aplicar tema moderno
    apply_modern_theme()
    
    # Criar cabeçalho
    create_header(
        title="HydroSim-RF Design System",
        subtitle="Showcase dos Componentes Visuais",
        logo_main_path=None,  # Adicionar caminho se tiver logo
        logo_secondary_path=None
    )
    
    # ========== SEÇÃO 1: CARDS DE MÉTRICAS ==========
    create_section_divider("1. Cards de Métricas")
    
    st.write("Cards individuais com diferentes status:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_metric_card(
            label="Status Normal",
            value="2,450",
            unit="células",
            status="default",
            icon="📊"
        )
    
    with col2:
        create_metric_card(
            label="Status Sucesso",
            value="100%",
            unit="completo",
            status="success",
            icon="✅"
        )
    
    with col3:
        create_metric_card(
            label="Status Crítico",
            value="92.5",
            unit="km²",
            status="danger",
            icon="🚨"
        )
    
    st.write("")
    
    # ========== SEÇÃO 2: LINHAS DE MÉTRICAS ==========
    create_section_divider("2. Linhas de Métricas (Responsivas)")
    
    metrics = [
        {
            'label': 'Tempo de Simulação',
            'value': '2',
            'unit': 'h 45m',
            'status': 'success',
            'icon': '⏱️'
        },
        {
            'label': 'Área Inundada',
            'value': '125.5',
            'unit': 'km²',
            'status': 'warning',
            'icon': '💧'
        },
        {
            'label': 'Volume de Água',
            'value': '2.3',
            'unit': 'M m³',
            'status': 'danger',
            'icon': '🌊'
        },
        {
            'label': 'Taxa de Precisão (RF)',
            'value': '94.2',
            'unit': '%',
            'status': 'success',
            'icon': '🤖'
        },
    ]
    
    create_metric_row(metrics)
    
    st.write("")
    
    # ========== SEÇÃO 3: INFO BOXES ==========
    create_section_divider("3. Caixas de Informação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_info_box(
            title="Informação",
            content="Este é um exemplo de caixa de informação. Use para dicas e contexto geral.",
            info_type="info",
            icon="ℹ️"
        )
        
        create_info_box(
            title="Sucesso!",
            content="A simulação foi executada com sucesso em 2h 45 minutos!",
            info_type="success",
            icon="✅"
        )
    
    with col2:
        create_info_box(
            title="Atenção",
            content="Seu DEM é grande (5000x5000px). Considere usar fator de reamostragem.",
            info_type="warning",
            icon="⚠️"
        )
        
        create_info_box(
            title="Erro",
            content="Falha ao carregar arquivo GeoTIFF. Verifique o formato.",
            info_type="error",
            icon="❌"
        )
    
    st.write("")
    
    # ========== SEÇÃO 4: GRADE DE ESTATÍSTICAS ==========
    create_section_divider("4. Grade de Estatísticas")
    
    stats = {
        'Resolução DEM': '30m',
        'Células da Grade': '1.2M',
        'Tempo de Passo': '60s',
        'Iterações': '2,400',
        'Memória Usada': '850MB',
        'CPU Médio': '65%',
    }
    
    create_stats_grid(stats)
    
    st.write("")
    
    # ========== SEÇÃO 5: TIMELINE DE PROGRESSO ==========
    create_section_divider("5. Timeline de Progresso")
    
    st.write("**Estado 1: Simulação em progresso**")
    steps_active = [
        {'name': 'Preparação', 'status': 'complete'},
        {'name': 'Simulação', 'status': 'active'},
        {'name': 'Pós-processamento', 'status': 'pending'},
        {'name': 'Exportação', 'status': 'pending'},
    ]
    create_progress_timeline(steps_active)
    
    st.write("**Estado 2: Simulação concluída**")
    steps_complete = [
        {'name': 'Preparação', 'status': 'complete'},
        {'name': 'Simulação', 'status': 'complete'},
        {'name': 'Pós-processamento', 'status': 'complete'},
        {'name': 'Exportação', 'status': 'complete'},
    ]
    create_progress_timeline(steps_complete)
    
    st.write("")
    
    # ========== SEÇÃO 6: EXEMPLO DE USO NO APP ==========
    create_section_divider("6. Exemplo de Integração Prática")
    
    with st.expander("Ver código de integração", expanded=False):
        st.code("""
from design import create_header, create_metric_row, create_info_box

def main():
    apply_modern_theme()
    
    # Cabeçalho
    create_header(
        title="HydroSim-RF",
        subtitle="Hybrid Simulator de Floods"
    )
    
    # Simulação...
    simulation_results = run_simulation()
    
    # Exibir métricas em cards modernos
    metrics = [
        {
            'label': f"Tempo: {simulation_results['time']}",
            'value': '2h 45m',
            'unit': '',
            'status': 'success',
            'icon': '⏱️'
        },
        {
            'label': 'Área Inundada',
            'value': f"{simulation_results['area']:.1f}",
            'unit': 'km²',
            'status': 'warning',
            'icon': '💧'
        },
    ]
    create_metric_row(metrics)
    
    # Info box com resultado
    create_info_box(
        title="Simulação Concluída",
        content="Análise RF iniciada...",
        info_type="success",
        icon="✅"
    )
        """, language="python")
    
    st.write("")
    
    # ========== SIDEBAR ==========
    st.sidebar.title("⚙️ Configurações")
    st.sidebar.write("Customize o design conforme necessário.")
    
    theme = st.sidebar.selectbox(
        "Tema",
        ["Moderno (Padrão)", "Clássico", "Escuro (Beta)"]
    )
    
    st.sidebar.write(f"**Tema selecionado**: {theme}")
    
    st.sidebar.markdown("---")
    st.sidebar.write("### 📚 Documentação")
    st.sidebar.write("""
    - [README](./README.md)
    - [Guia de Integração](./DESIGN_INTEGRATION.md)
    - [Código do Design](./design.py)
    """)


if __name__ == "__main__":
    main()
