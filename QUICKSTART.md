# HydroSim-RF - Guia Rápido de Início

## 📋 O que foi criado?

Seu projeto está totalmente configurado com:

✅ **Virtual Environment (venv)** - Ambiente Python isolado  
✅ **Docker & Docker Compose** - Containerização completa  
✅ **Requirements.txt** - Todas as dependências catalogadas  
✅ **Documentação** - README detalhado  
✅ **Scripts de Inicialização** - Facilitar execução  
✅ **Makefile** - Comandos simplificados  
✅ **Configurações** - Streamlit, Docker, ambiente

---

## 🚀 Como Iniciar

### **Opção 1: Local com venv (Recomendado para desenvolvimento)**

```bash
cd /home/leticia/Desktop/hydrosim

# Método A: Usar Makefile
make run

# Método B: Script automatizado
./start.sh

# Método C: Manual
source venv/bin/activate
pip install -r requirements.txt
streamlit run hydrosim_rf.py
```

A aplicação estará em: **[http://localhost:8501](http://localhost:8501)**

---

### **Opção 2: Docker (Recomendado para produção)**

```bash
cd /home/leticia/Desktop/hydrosim

# Iniciar com docker-compose
make docker-up

# Ou manual
docker-compose up -d

# Ver logs
make docker-logs
```

A aplicação estará em: **[http://localhost:8501](http://localhost:8501)**

---

## 🔍 Verificar Instalação

```bash
# Verificar se tudo está certo
python3 check_installation.py

# Ou com make
make install
```

---

## 📁 Estrutura Criada

```text
hydrosim/
├── hydrosim_rf.py              # App principal
├── shapes.py                   # Estilos
├── requirements.txt            # Dependências Python
├── Dockerfile                  # Imagem Docker
├── docker-compose.yml          # Orquestração produção
├── docker-compose.dev.yml      # Orquestração desenvolvimento
├── Makefile                    # Comandos úteis
├── start.sh                    # Script de inicialização
├── check_installation.py       # Verificador de instalação
├── README.md                   # Documentação completa
├── .env.example                # Variáveis de ambiente
├── .gitignore                  # Git ignore
├── .dockerignore               # Docker ignore
├── .streamlit/                 # Configurações Streamlit
│   └── config.toml
├── venv/                       # Ambiente virtual Python
├── data/
│   ├── input/                  # Arquivos de entrada (DEMs, shapefiles)
│   └── output/                 # Resultados de simulações
└── logs/                       # Arquivos de log
```

---

## 📦 Comandos Úteis

### **Com Makefile**

```bash
make help          # Ver todos os comandos
make install       # Instalar dependências
make run           # Executar localmente
make docker-build  # Construir imagem Docker
make docker-up     # Iniciar container
make docker-down   # Parar container
make clean         # Limpar arquivos temporários
```

### **Com Docker Compose**

```bash
docker-compose up -d           # Iniciar (background)
docker-compose down            # Parar
docker-compose logs -f         # Ver logs em tempo real
docker-compose ps              # Ver status dos serviços
docker-compose restart         # Reiniciar
```

---

## 🔧 Solução de Problemas

### **"Module not found" errors**

```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### **Porta 8501 já em uso**

```bash
# Mudar porta no docker-compose.yml ou:
streamlit run hydrosim_rf.py --server.port=8502
```

### **Problemas com GDAL**

```bash
# Linux/Ubuntu
sudo apt-get install gdal-bin libgdal-dev libproj-dev libgeos-dev

# macOS
brew install gdal proj geos

# Use Docker se continuar com problemas
make docker-up
```

### **Container não inicia**

```bash
# Ver logs detalhados
docker-compose logs hydrosim-app

# Reconstruir imagem
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

## 📊 Usando a Aplicação

### **Passo 1: Preparar Dados**

- Ter um **DEM em GeoTIFF** (obrigatório)
- Opcionalmente: shapefiles/GeoPackage com polígonos de chuva ou rio

### **Passo 2: Upload**

1. Abrir [http://localhost:8501](http://localhost:8501)
2. Fazer upload dos arquivos
3. Configurar parâmetros de simulação

### **Passo 3: Executar**

- Clicar "Run Simulation"
- Acompanhar em tempo real
- Exportar resultados (CSV, PNG, GIF, ZIP)

### **Passo 4: Análise IA**

- Treinar Random Forest com "Train RF Classifier"
- Gerar mapa de probabilidade
- Validar modelo

### **Passo 5: Mitigação**

- Executar "Run Mitigation Analysis"
- Gerar relatório com sugestões de intervenção

---

## 🌍 Variáveis de Ambiente

Criar arquivo `.env`:

```bash
cp .env.example .env
```

Editar conforme necessário:

```env
STREAMLIT_SERVER_PORT=8501
ENABLE_LISFLOOD=false
PYTHONUNBUFFERED=1
```

---

## 📚 Recursos Adicionais

- **Documentação completa**: Ver `README.md`
- **Código-fonte**: `hydrosim_rf.py`
- **Configurações Streamlit**: `.streamlit/config.toml`
- **Dependências**: `requirements.txt`

---

## ⚡ Performance

- **DEMs pequenos (< 1000x1000 px)**: Rápido em qualquer máquina
- **DEMs grandes**: Usar "Grid resampling factor" = 4, 8 ou 16
- **Animações**: Desabilitar "Quick preview" para economizar tempo
- **Docker**: Melhor para máquinas com recursos limitados

---

## 🐳 Desenvolvimento com Docker

Para trabalhar com live reloading:

```bash
docker-compose -f docker-compose.dev.yml up
```

Isso monta o diretório atual como volume, permitindo edição em tempo real.

---

## 📝 Próximos Passos

1. **Testar instalação**: `python3 check_installation.py`
2. **Fazer upload de dados**: Preparar DEM em GeoTIFF
3. **Executar simulação**: Testar parâmetros diferentes
4. **Validar com IA**: Treinar modelo Random Forest
5. **Gerar relatórios**: Exportar dados para análise

---

## 🆘 Suporte

Se encontrar problemas:

1. Verifique os logs: `docker-compose logs`
2. Rode o verificador: `python3 check_installation.py`
3. Consulte o `README.md` completo
4. Verifique a documentação do Streamlit

---

**Pronto para começar!** 🎯

```bash
cd /home/leticia/Desktop/hydrosim
make run
```

Ou com Docker:

```bash
make docker-up
```

---

Versão: 1.0.0  
Desenvolvedor: Letícia Caldas  
Data: Março 2026
