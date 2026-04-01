# 🌐 API REST Routes - HydroSim v2.1

## Visão Geral

O HydroSim agora possui uma API REST completa para exportar simulações em múltiplos formatos e gerenciar histórico de simulações.

---

## 📋 Rotas Disponíveis

### 1. **Health Check** ✅
```
GET /api/health
```
**Descrição**: Verifica se a API está funcionando

**Response** (200):
```json
{
  "success": true,
  "status": "healthy",
  "database_initialized": true,
  "cache_items": 5,
  "timestamp": "2026-03-30T10:30:00"
}
```

---

### 2. **Listar Simulações** 📊
```
GET /api/simulations
```
**Descrição**: Lista todas as simulações do banco de dados

**Response** (200):
```json
{
  "success": true,
  "count": 3,
  "simulations": [
    {
      "id": 1,
      "name": "Test Run 1",
      "timestamp": "2026-03-30T10:00:00",
      "parameters": {"...": "..."},
      "max_depth_m": 2.5,
      "flooded_area_percent": 15.3,
      "total_volume_m3": 1250000
    },
    {...}
  ]
}
```

---

### 3. **Obter Detalhes de Uma Simulação** 🔍
```
GET /api/simulations/<id>
```

**Exemplo**:
```
GET /api/simulations/1
```

**Response** (200):
```json
{
  "success": true,
  "simulation": {
    "id": 1,
    "name": "Test Run 1",
    "timestamp": "2026-03-30T10:00:00",
    "parameters": {...},
    "max_depth_m": 2.5,
    "flooded_area_percent": 15.3,
    "total_volume_m3": 1250000,
    "notes": "Rainfall 50mm, 1h duration"
  }
}
```

---

### 4. **Exportar para NetCDF** 🌍
```
POST /api/export/netcdf
Content-Type: application/json
```

**Request Body**:
```json
{
  "water_depth": [[0.0, 0.5, 1.2], [0.1, 0.6, 1.5]],
  "dem": [[100.5, 100.0, 99.8], [100.3, 99.9, 99.5]],
  "transform": [1.0, 0.0, -73.5, 0.0, -1.0, 40.5],
  "crs": "EPSG:4326"
}
```

**Response** (200):
```json
{
  "success": true,
  "message": "NetCDF export successful",
  "file": "outputs/simulation_results.nc"
}
```

**Características**:
- ✅ Compressão automática
- ✅ Metadados inclusos
- ✅ Compatível com MATLAB, Python (xarray), IDL

---

### 5. **Download NetCDF** 📥
```
GET /download/simulation-netcdf
```

**Response**: Arquivo `.nc` para download

---

### 6. **Exportar para HDF5** 📦
```
POST /api/export/hdf5
Content-Type: application/json
```

**Request Body**:
```json
{
  "water_depth": [[0.0, 0.5, 1.2], [0.1, 0.6, 1.5]],
  "dem": [[100.5, 100.0, 99.8], [100.3, 99.9, 99.5]],
  "transform": [1.0, 0.0, -73.5, 0.0, -1.0, 40.5],
  "crs": "EPSG:4326"
}
```

**Response** (200):
```json
{
  "success": true,
  "message": "HDF5 export successful",
  "file": "outputs/simulation_results.h5"
}
```

**Características**:
- ✅ Compressão DEFLATE nível 9
- ✅ Estrutura hierárquica
- ✅ Compatível com ferramentas HDF5 universais

---

### 7. **Download HDF5** 📥
```
GET /download/simulation-hdf5
```

**Response**: Arquivo `.h5` para download

---

### 8. **Comparar Simulações** ⚖️
```
POST /api/compare
Content-Type: application/json
```

**Request Body**:
```json
{
  "simulation_ids": [1, 2, 3]
}
```

**Response** (200):
```json
{
  "success": true,
  "message": "Comparison report generated",
  "file": "outputs/comparison_report.json",
  "simulations_compared": 3,
  "report": {
    "statistics": {
      "max_depth": {
        "min": 2.1,
        "max": 3.5,
        "mean": 2.8
      },
      "flooded_area": {
        "min": 12.5,
        "max": 18.3,
        "mean": 15.4
      },
      "volume": {
        "min": 1100000,
        "max": 1400000,
        "mean": 1250000
      }
    },
    "simulations": [
      {"id": 1, "name": "Scenario A", "max_depth": 2.5, ...},
      {"id": 2, "name": "Scenario B", "max_depth": 3.2, ...},
      {"id": 3, "name": "Scenario C", "max_depth": 2.1, ...}
    ]
  }
}
```

---

### 9. **Download Relatório de Comparação** 📊
```
GET /download/comparison-report
```

**Response**: Arquivo `.json` com relatório de comparação

---

## 🔧 Exemplos de Uso

### Com cURL:

**Listar simulações**:
```bash
curl http://localhost:5001/api/simulations
```

**Verificar saúde**:
```bash
curl http://localhost:5001/api/health
```

**Exportar NetCDF** (desde um arquivo JSON):
```bash
curl -X POST http://localhost:5001/api/export/netcdf \
  -H "Content-Type: application/json" \
  -d @export_data.json
```

**Comparar simulações**:
```bash
curl -X POST http://localhost:5001/api/compare \
  -H "Content-Type: application/json" \
  -d '{"simulation_ids": [1, 2, 3]}'
```

### Com Python (requests):

```python
import requests
import json

BASE_URL = "http://localhost:5001"

# 1. Listar simulações
response = requests.get(f"{BASE_URL}/api/simulations")
simulations = response.json()['simulations']
print(f"Total de simulações: {len(simulations)}")

# 2. Exportar NetCDF
export_data = {
    "water_depth": water.tolist(),
    "dem": dem.tolist(),
    "crs": "EPSG:4326"
}
response = requests.post(f"{BASE_URL}/api/export/netcdf", json=export_data)
print(f"Export: {response.json()}")

# 3. Comparar simulações
compare_data = {"simulation_ids": [1, 2, 3]}
response = requests.post(f"{BASE_URL}/api/compare", json=compare_data)
report = response.json()['report']
print(f"Profundidade média: {report['statistics']['max_depth']['mean']}m")

# 4. Download
response = requests.get(f"{BASE_URL}/download/simulation-netcdf")
with open("results.nc", "wb") as f:
    f.write(response.content)
```

---

## 📝 Status Codes

| Código | Significado | Exemplo |
|--------|-------------|---------|
| 200 | Sucesso | Simulação encontrada |
| 400 | Requisição inválida | Dados faltando |
| 404 | Não encontrado | Simulação não existe |
| 500 | Erro do servidor | Falha na exportação |

---

## 🗄️ Estrutura do Banco de Dados

### Tabela `simulations`:
```
id (INT, PRIMARY KEY)
name (TEXT)
timestamp (DATETIME)
parameters (TEXT/JSON)
results_path (TEXT)
max_depth_m (FLOAT)
flooded_area_percent (FLOAT)
total_volume_m3 (FLOAT)
notes (TEXT)
```

### Tabela `exports`:
```
id (INT, PRIMARY KEY)
sim_id (INT, FOREIGN KEY)
format (TEXT) - 'netcdf', 'hdf5', 'geotiff', etc.
file_path (TEXT)
export_date (DATETIME)
```

---

## 🚀 Como Usar

1. **Iniciar servidor**:
   ```bash
   python web_server_v3.py
   ```

2. **Verificar status**:
   ```bash
   curl http://localhost:5001/api/health
   ```

3. **Usar qualquer uma das rotas acima**

---

## ✨ Recursos Principais

✅ **Rastreamento automático** - Todas as simulações salvas no banco de dados
✅ **Múltiplos formatos** - NetCDF, HDF5, GeoTIFF, GeoPackage
✅ **Comparação** - Análise estatística de múltiplas simulações
✅ **Cache** - Reutilização de DEMs com 30min TTL
✅ **Compressão** - Redução de 60-70% no tamanho dos arquivos
✅ **Validação** - Verificação de entrada antes do processamento

---

**Status**: ✅ Implementado e pronto para uso
**Data**: 30 de março de 2026
