"""
HydroSim-RF Custom Styles Module

This module fornece CSS styles customized para a interface Streamlit
e funções de layout para a aplicação HydroSim-RF.

Author: Letícia Caldas
License: MIT
"""

from typing import Optional
import streamlit as st


def apply_custom_styles():
    """
    Aplica CSS styles customized à interface Streamlit.
    
    Inclui:
    - Cartões personalizados (custom-card)
    - Paleta de cores consistente
    - Tipografia melhorada
    - Responsividade
    """
    st.markdown("""
    <style>
    /* Estilos globais */
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00cc66;
        --danger-color: #cc0000;
        --warning-color: #ffaa00;
        --info-color: #0099cc;
    }
    
    /* Cartões personalizados */
    .custom-card {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .custom-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: box-shadow 0.3s ease;
    }
    
    /* Estilos de header */
    h1 {
        color: #0066cc;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #0066cc;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #333;
        font-weight: 500;
    }
    
    /* Estilos de tabelas */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    th {
        background-color: #0066cc;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    
    td {
        padding: 10px 12px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    tr:hover {
        background-color: #f5f5f5;
    }
    
    /* Estilos de botões */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
    }
    
    /* Estilos de entrada */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
        font-size: 14px;
    }
    
    /* Estilos de alerta */
    .stAlert {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Código */
    code {
        background-color: #f0f0f0;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    
    pre {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 8px;
        overflow-x: auto;
    }
    
    </style>
    """, unsafe_allow_html=True)


def create_header(title: str, subtitle: str = "", logo_path: Optional[str] = None):
    """
    Cria um header personalizado para a aplicação.
    
    Parameters
    ----------
    title : str
        Título principal
    subtitle : str, optional
        Subtítulo
    logo_path : str, optional
        Caminho para logo (se existir)
    
    Returns
    -------
    None
    """
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        if logo_path:
            try:
                st.image(logo_path, width=100)
            except Exception:
                pass
    
    with col_title:
        st.markdown(f"# {title}")
        if subtitle:
            st.markdown(f"**{subtitle}**")
    
    st.markdown("---")
