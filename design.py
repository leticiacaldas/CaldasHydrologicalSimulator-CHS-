"""
design.py - Componentes de Design Moderno para HydroSim-RF

Módulo que fornece componentes visuais customizados, estatísticas
e layouts responsivos para a interface do simulador de inundações.

Author: Letícia Caldas
License: MIT
"""

from typing import Optional, Dict, Any, List
import streamlit as st
import base64
from pathlib import Path


def _path_to_data_uri(path: str) -> str:
    """Converte um caminho de imagem local para data URI base64."""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        ext = Path(path).suffix.lower().replace(".", "")
        mime = "image/png" if ext == "png" else "image/jpeg" if ext in ["jpg", "jpeg"] else "image/svg+xml"
        return f"data:{mime};base64,{b64}"
    except (OSError, ValueError):
        return path


def apply_modern_theme():
    """Aplica tema moderno e responsivo ao Streamlit."""
    st.markdown("""
    <style>
    /* ===== CORES E VARIÁVEIS ===== */
    :root {
        --primary: #1e88e5;
        --secondary: #26c6da;
        --accent: #ff6f00;
        --success: #43a047;
        --warning: #fbc02d;
        --danger: #e53935;
        --dark-text: #1a1a1a;
        --light-bg: #f5f7fa;
        --border-color: #e0e0e0;
    }
    
    /* ===== RESET E BASE ===== */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: var(--light-bg);
        color: var(--dark-text);
    }
    
    .main {
        background-color: var(--light-bg);
        padding: 0;
    }
    
    /* ===== CABEÇALHO ===== */
    .header-container {
        background: linear-gradient(135deg, #1e88e5 0%, #26c6da 100%);
        color: white;
        padding: 2rem;
        border-radius: 0 0 16px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 2rem;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.15);
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .logo-container {
        flex: 0 0 auto;
    }
    
    .logo-main {
        height: 80px;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .logo-main:hover {
        transform: scale(1.05);
    }
    
    .logo-main img {
        height: 100%;
        width: auto;
        object-fit: contain;
    }
    
    .title-section {
        flex: 1 1 auto;
        text-align: center;
        min-width: 250px;
    }
    
    .title-section h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .title-section p {
        font-size: 0.95rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .logo-secondary {
        flex: 0 0 auto;
        height: 60px;
        opacity: 0.8;
        transition: opacity 0.3s ease;
    }
    
    .logo-secondary:hover {
        opacity: 1;
    }
    
    .logo-secondary img {
        height: 100%;
        width: auto;
        object-fit: contain;
    }
    
    /* ===== TIPOGRAFIA ===== */
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #1e88e5, #26c6da);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-size: 1.75rem;
        border-bottom: 3px solid #26c6da;
        padding-bottom: 0.75rem;
    }
    
    h3 {
        font-size: 1.25rem;
        color: #1e88e5;
    }
    
    /* ===== BOTÕES ===== */
    .stButton > button {
        background: linear-gradient(135deg, #1e88e5, #26c6da);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.25);
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(30, 136, 229, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ===== ABAS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid var(--border-color);
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #1e88e5 !important;
        border-bottom: 3px solid #1e88e5 !important;
    }
    
    /* ===== CARDS ===== */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1e88e5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card.success {
        border-left-color: #43a047;
    }
    
    .metric-card.warning {
        border-left-color: #fbc02d;
    }
    
    .metric-card.danger {
        border-left-color: #e53935;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #999;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    .metric-unit {
        font-size: 0.9rem;
        color: #999;
        margin-left: 0.25rem;
    }
    
    /* ===== INPUTS E SLIDERS ===== */
    .stNumberInput input, .stTextInput input, .stSelectbox input {
        border: 2px solid var(--border-color) !important;
        border-radius: 8px;
        padding: 0.75rem !important;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus, .stTextInput input:focus, .stSelectbox input:focus {
        border-color: #1e88e5 !important;
        box-shadow: 0 0 0 3px rgba(30, 136, 229, 0.1) !important;
    }
    
    .stSlider [data-testid="stThumb"] {
        background-color: #1e88e5;
    }
    
    .stSlider [data-testid="stTickBar"] {
        background: linear-gradient(90deg, #1e88e5, #26c6da);
    }
    
    /* ===== UPLOAD ===== */
    .stFileUploader > div > div {
        border: 2px dashed #1e88e5;
        border-radius: 12px;
        background-color: rgba(30, 136, 229, 0.05);
        transition: all 0.3s ease;
        padding: 2rem;
    }
    
    .stFileUploader > div > div:hover {
        background-color: rgba(30, 136, 229, 0.1);
        border-color: #26c6da;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1e88e5, #26c6da);
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background-color: #f5f7fa;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-weight: 600;
        color: #1e88e5;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(30, 136, 229, 0.05);
    }
    
    /* ===== TABELAS ===== */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden;
    }
    
    .dataframe thead tr {
        background-color: #f5f7fa;
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, #1e88e5, #26c6da);
        color: white;
        font-weight: 600;
        padding: 1rem !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(30, 136, 229, 0.05);
    }
    
    .dataframe tbody td {
        padding: 0.75rem !important;
    }
    
    /* ===== SIDEBAR ===== */
    .css-1d391kg {
        background-color: white;
        border-right: 1px solid var(--border-color);
    }
    
    /* ===== NOTIFICAÇÕES ===== */
    .stAlert {
        border-radius: 8px;
    }
    
    .stAlert > div > div {
        padding: 1rem;
    }
    
    /* ===== RESPONSIVO ===== */
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column;
            padding: 1.5rem;
            gap: 1rem;
        }
        
        .title-section h1 {
            font-size: 1.5rem;
        }
        
        h2 {
            font-size: 1.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
    
    /* ===== ANIMAÇÕES ===== */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    .stProgress > div {
        animation: pulse 2s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)


def create_header(
    title: str = "HydroSim-RF",
    subtitle: str = "Simulador Híbrido de Inundações",
    logo_main_path: Optional[str] = None,
    logo_secondary_path: Optional[str] = None,
):
    """
    Cria um cabeçalho profissional com logos e títulos.
    
    Parameters
    ----------
    title : str
        Título principal
    subtitle : str
        Subtítulo descritivo
    logo_main_path : str, optional
        Caminho para a logo principal
    logo_secondary_path : str, optional
        Caminho para a logo secundária (instituição/projeto)
    """
    logo_main_uri = _path_to_data_uri(logo_main_path) if logo_main_path else ""
    logo_secondary_uri = _path_to_data_uri(logo_secondary_path) if logo_secondary_path else ""
    
    html = '<div class="header-container">'
    
    if logo_main_uri:
        html += f'<div class="logo-container"><div class="logo-main"><img src="{logo_main_uri}" alt="Logo Principal"></div></div>'
    
    html += f"""
    <div class="title-section">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    
    if logo_secondary_uri:
        html += f'<div class="logo-secondary"><img src="{logo_secondary_uri}" alt="Logo Secundária"></div>'
    
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)


def create_metric_card(
    label: str,
    value: str,
    unit: str = "",
    status: str = "default",
    icon: str = "📊",
):
    """
    Cria um card de métrica com ícone, rótulo e valor.
    
    Parameters
    ----------
    label : str
        Rótulo da métrica
    value : str
        Valor principal
    unit : str
        Unidade da medida
    status : str
        Status visual: 'default', 'success', 'warning', 'danger'
    icon : str
        Ícone emoji para exibir
    """
    status_class = f' {status}' if status != 'default' else ''
    
    html = f"""
    <div class="metric-card{status_class}">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <span class="metric-label">{label}</span>
        </div>
        <div>
            <span class="metric-value">{value}</span>
            <span class="metric-unit">{unit}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_metric_row(metrics: List[Dict[str, str]]):
    """
    Cria uma linha de cards de métricas em coluna.
    
    Parameters
    ----------
    metrics : list
        Lista de dicts com 'label', 'value', 'unit', 'status', 'icon'
    
    Example
    -------
    metrics = [
        {'label': 'Área Inundada', 'value': '125.5', 'unit': 'km²', 'icon': '💧'},
        {'label': 'Volume de Água', 'value': '2.3M', 'unit': 'm³', 'icon': '🌊'},
        {'label': 'Tempo de Simulação', 'value': '2h 45m', 'unit': '', 'icon': '⏱️'}
    ]
    create_metric_row(metrics)
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            create_metric_card(
                label=metric.get('label', ''),
                value=metric.get('value', '0'),
                unit=metric.get('unit', ''),
                status=metric.get('status', 'default'),
                icon=metric.get('icon', '📊'),
            )


def create_section_divider(title: str = ""):
    """Cria um divisor visual com título opcional."""
    if title:
        st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<hr style='border: none; border-top: 3px solid #26c6da; margin: 2rem 0;'>", unsafe_allow_html=True)


def create_info_box(
    title: str,
    content: str,
    info_type: str = "info",
    icon: str = "ℹ️",
):
    """
    Cria uma caixa de informação estilizada.
    
    Parameters
    ----------
    title : str
        Título da caixa
    content : str
        Conteúdo/descrição
    info_type : str
        Tipo: 'info', 'success', 'warning', 'error'
    icon : str
        Ícone emoji
    """
    colors = {
        'info': '#1e88e5',
        'success': '#43a047',
        'warning': '#fbc02d',
        'error': '#e53935',
    }
    
    color = colors.get(info_type, colors['info'])
    
    html = f"""
    <div style="
        background-color: rgba({color}, 0.05);
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    ">
        <div style="display: flex; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.25rem;">{icon}</span>
            <strong style="color: {color};">{title}</strong>
        </div>
        <p style="margin: 0; color: #666;">{content}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_stats_grid(stats: Dict[str, Any]):
    """
    Cria uma grade de estatísticas lado a lado.
    
    Parameters
    ----------
    stats : dict
        Dicionário com labels como chaves e valores como valores
    
    Example
    -------
    stats = {
        'Resolução DEM': '30m',
        'Células da Grade': '1.2M',
        'Tempo de Passo': '60s',
        'Iterações': '2400',
    }
    create_stats_grid(stats)
    """
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats.items()):
        with col:
            st.metric(label=label, value=value)


def create_progress_timeline(steps: List[Dict[str, str]]):
    """
    Cria uma timeline de progresso com múltiplos passos.
    
    Parameters
    ----------
    steps : list
        Lista de dicts com 'name', 'status' ('pending', 'active', 'complete')
    """
    html = '<div style="display: flex; align-items: center; gap: 1rem; margin: 2rem 0;">'
    
    for i, step in enumerate(steps):
        status = step.get('status', 'pending')
        colors = {
            'pending': '#ccc',
            'active': '#1e88e5',
            'complete': '#43a047'
        }
        color = colors.get(status, colors['pending'])
        icons = {
            'pending': '○',
            'active': '◐',
            'complete': '✓'
        }
        icon = icons.get(status, '○')
        
        html += f"""
        <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
            <div style="
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background-color: {color};
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-bottom: 0.5rem;
            ">
                {icon}
            </div>
            <small style="text-align: center; color: #666;">{step.get('name', 'Passo')}</small>
        </div>
        """
        
        if i < len(steps) - 1:
            html += f"""
            <div style="
                height: 2px;
                background-color: {color};
                flex: 1;
                margin-bottom: 2rem;
            "></div>
            """
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    # Exemplo de uso
    apply_modern_theme()
    create_header(
        title="HydroSim-RF",
        subtitle="Simulador Híbrido de Inundações Urbanas"
    )
    
    st.write("## Exemplo de Cards de Métricas")
    metrics = [
        {'label': 'Área Inundada', 'value': '125.5', 'unit': 'km²', 'status': 'warning', 'icon': '💧'},
        {'label': 'Volume', 'value': '2.3M', 'unit': 'm³', 'status': 'danger', 'icon': '🌊'},
        {'label': 'Tempo', 'value': '2h 45m', 'unit': '', 'status': 'success', 'icon': '⏱️'}
    ]
    create_metric_row(metrics)
    
    st.write("## Info Boxes")
    create_info_box(
        "Dica de Performance",
        "Use o fator de reamostragem 4x para DEMs maiores que 5000x5000 pixels.",
        "info"
    )
