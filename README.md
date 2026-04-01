# HydroSim-RF: Hybrid Raster-Based Urban Flood Simulation Framework

## Descrição

**HydroSim-RF** é uma aplicação web interativa para simulação rápida de inundações urbanas em 2D usando Modelos de Elevação Digital (DEMs). Implementa:

- **Núcleo hidrodinâmico**: Aproximação de onda de difusão via solver NumPy vetorizado (`DiffusionWaveFloodModel`)
- **Classificador de Machine Learning**: Random Forest para estimativa de probabilidade de inundação sem necessidade de dados de calibração
- **Análise espacial**: Identificação automática de zonas elegíveis para mitigação de inundações (reflorestamento, diques, drenagem, aterro)
- **Visualização interativa**: Streamlit com basemaps online, animações (GIF/MP4) e exportação de dados

## Requisitos do Sistema

### Local (com venv)

- **Python**: 3.9+
- **GDAL**: Sistema operacional (Linux/macOS/Windows)
- **Memoria**: 4GB recomendado
- **Processador**: Qualquer processador moderno

### Docker

- **Docker**: 20.10+
- **Docker Compose**: 1.29+
- **Disco**: 2-3GB para a imagem

## Instalação

### Opção 1: Ambiente Virtual Local (venv)

```bash
cd /home/leticia/Desktop/hydrosim

# Ativar ambiente virtual
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
streamlit run hydrosim_rf.py
```

A aplicação estará disponível em: **[http://localhost:8501](http://localhost:8501)**

### Opção 2: Docker (Recomendado)

```bash
cd /home/leticia/Desktop/hydrosim

# Construir imagem
docker build -t hydrosim-rf .

# Executar com docker-compose
docker-compose up -d

# Ou executar diretamente
docker run -p 8501:8501 -v $(pwd)/data:/app/data hydrosim-rf
```

**A aplicação estará disponível em**: [http://localhost:8501](http://localhost:8501)

## Estrutura do Projeto

```text
hydrosim/
├── hydrosim_rf.py              # Aplicação principal (Streamlit)
├── shapes.py                   # Estilos CSS customizados
├── requirements.txt            # Dependências Python
├── Dockerfile                  # Configuração Docker
├── docker-compose.yml          # Orquestração de contêiner
├── .dockerignore               # Arquivos excluídos do Docker
├── .gitignore                  # Arquivos excluídos do Git
├── README.md                   # Este arquivo
├── logos/                      # Assets (logo, ícones)
├── data/
│   ├── input/                  # DEMs e vetores de entrada
│   └── output/                 # Resultados de simulações
├── logs/                       # Arquivos de log
└── venv/                       # Ambiente virtual Python
```

## Dependências Principais

| Pacote | Versão | Propósito |
| --- | --- | --- |
| `streamlit` | ≥1.28.0 | Framework web interativo |
| `numpy` | ≥1.24.0 | Computação numérica (solver) |
| `rasterio` | ≥1.3.0 | E/S de rasters geoespaciais |
| `geopandas` | ≥0.13.0 | Processamento vetorial |
| `scikit-learn` | ≥1.3.0 | Random Forest classifier |
| `matplotlib` | ≥3.7.0 | Visualização e animações |
| `imageio-ffmpeg` | ≥0.4.8 | Exportação de vídeos MP4 |

## Como Usar

### 1. Preparar Dados de Entrada

Você precisará de:

- **DEM (GeoTIFF)**: Raster de elevação em coordenadas geográficas
- **Vetor de Fontes (opcional)**: Polígonos (GeoPackage/.shp) definindo áreas de chuva
- **Vetor de Rio (opcional)**: Polígonos/linhas definindo a rede de drenagem
- **DOM/Ortoimage (opcional)**: Ortofoto para visualização

### 2. Executar Simulação

1. Abrir a aplicação: [http://localhost:8501](http://localhost:8501)
2. **Aba "Simulation"**:
   - Upload do DEM (obrigatório)
   - Upload de polígonos de fonte (opcional)
   - Configurar parâmetros de chuva, tempo de passo, coeficiente de difusão
   - Clicar em **"Run Simulation"**
3. A animação será gerada em tempo real

### 3. Validação com Random Forest (IA)

1. Após simulação, na seção **"Random Forest Inundation Probability"**:
   - Treinar modelo com `n_estimators` e `max_depth` customizáveis
   - O modelo aprende a relação entre topografia (elevação, declividade) e inundação

### 4. Análise de Mitigação

1. Na seção **"Spatial Flood Mitigation Analysis"**:
   - Executar análise espacial automática
   - Identificar zonas elegíveis para:
     - **Reflorestamento / infraestrutura verde**
     - **Diques/levees**
     - **Sistemas de drenagem**
     - **Aterro de terreno**
2. Gerar relatório detalhado com sugestões

## Recursos Avançados

### Exportação de Dados

- **CSV**: Série temporal de diagnósticos (área inundada, volume, profundidade)
- **GeoTIFF**: Rasters de probabilidade (IA) e inundação simulada
- **PNG**: Sobreposições de mapa
- **GIF/MP4**: Animações de inundação
- **ZIP**: Pacote completo com todos os artefatos

### Personalização Visual

- **Basemap**: Escolher entre Esri, CartoDB, OpenStreetMap ou nenhum
- **Hillshade**: Sombreamento analítico do DEM
- **Transparência e cores**: Controlar visualização da água
- **Contornos**: Destacar limite de inundação

### Parâmetros Hidrodinâmicos

- **Coeficiente de difusão (α)**: 0.01–1.0 (determina propagação lateral)
- **Limiar de inundação (h\*)**: Profundidade mínima para classificação
- **Tamanho do passo de tempo**: 1–1440 minutos
- **Fator de reamostragem**: 1×–16× (velocidade vs. precisão)

## Troubleshooting

### "ModuleNotFoundError: No module named 'rasterio'"

```bash
source venv/bin/activate
pip install --upgrade rasterio geopandas
```

### "GDAL not found"

```bash
# Linux (Ubuntu/Debian)
sudo apt-get install gdal-bin libgdal-dev libproj-dev libgeos-dev

# macOS
brew install gdal proj geos

# Usar Docker se problemas persistirem
docker-compose up
```

### "AnimationError: no writer for format 'mp4'"

```bash
pip install imageio-ffmpeg
```

### Aplicação lenta (simulação de larga escala)

- Aumentar **"Grid resampling factor"** para 4, 8 ou 16
- Reduzir duração da animação ("Quick preview" para testes)
- Executar no Docker com limite de memória: `docker-compose.yml`

## Estrutura de Código

### Classes Principais

#### `DiffusionWaveFloodModel`

Núcleo da simulação. Implementa propagação de água em 2D com:

- Redistribuição de água para células vizinhas de menor elevação
- Conservação de volume de água
- Registro de diagnósticos temporais
- Suporte a máscaras de fonte e rio

#### `RandomForestClassifier` (scikit-learn)

Treinado em:

- **Entradas**: Elevation normalizada, slope normalizada (derivados do DEM)
- **Saída**: Probabilidade binária de inundação (0–1)
- **Uso**: Transferência para novos DEMs sem simulação completa

### Funções de Análise

- `_prepare_spatial_domain()`: Lê DEM, reamostra, rasteriza vetores
- `_identify_intervention_zones()`: Análise de mitigação (DBSCAN, morphological filters)
- `_train_flood_classifier()`: Treino do RF com class balancing
- `_predict_probability()`: Predição em novo DEM
- `_build_mitigation_report()`: Relatório estruturado com citações científicas

## Referências Bibliográficas

- **Hunter et al. (2005)**: Diffusion-wave formulation para modelagem de inundação
- **Breiman (2001)**: Random Forests (teoria do classificador)
- **Neal et al. (2012)**: Aproximação zero-inércia para hidrodinâmica
- **Rogger et al. (2017)**: Impacto de mudança de uso do solo em inundações
- **EU Directive 2007/60/EC**: Gestão de risco de inundação

## Licença

MIT License — Ver arquivo LICENSE para detalhes

## Suporte

Para problemas, abra uma issue no repositório GitHub ou entre em contato com a equipe de desenvolvimento.

---

**Versão**: 1.0.0  
**Último update**: Março 2026  
**Autor**: Letícia Caldas  
**Instituição**: HydroLab Research Group
