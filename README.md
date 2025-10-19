# Análisis de Señales EEG: FFT, STFT y CWT

Este proyecto implementa un análisis comparativo de señales EEG utilizando tres transformadas diferentes para identificar contenido en frecuencia:

1. **FFT** (Fast Fourier Transform) - Espectro de frecuencias
2. **STFT** (Short-Time Fourier Transform) - Espectrograma  
3. **CWT** (Continuous Wavelet Transform) - Escalograma

## 🎯 Objetivo

Responder la pregunta: **¿Qué contenido en frecuencia identifica cada transformada?**

## 📊 Archivos de Datos

- `FileEEG.mat`: Señal EEG de 1024 Hz, 180 segundos, 2 canales
- `sEEG.mat`: Señal EEG de 256 Hz, 78 segundos, 1 canal (T8-P8)

## 🚀 Instalación y Uso

### Requisitos
- Python 3.10+
- Archivos de datos EEG (.mat)

### Configuración del Ambiente

```bash
# Crear ambiente virtual
python3 -m venv venv

# Activar ambiente virtual
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate      # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar Análisis

```bash
python analysis.py
```

## 📈 Resultados

El script genera:

- **`dashboard.html`**: Dashboard interactivo con gráficas comparativas
- **`analysis_report.md`**: Reporte detallado del análisis

### Dashboard Interactivo

El dashboard incluye:
- Espectro FFT con picos identificados
- Espectrograma STFT (tiempo-frecuencia)
- Escalograma CWT (tiempo-frecuencia adaptativo)
- Comparación de tiempos de procesamiento

## 🔬 Metodología

### FFT (Fast Fourier Transform)
- **Propósito**: Identificar componentes frecuenciales de toda la señal
- **Ventaja**: Muy rápida, ideal para análisis inicial
- **Limitación**: Sin resolución temporal

### STFT (Short-Time Fourier Transform)
- **Propósito**: Análisis tiempo-frecuencia con ventanas fijas
- **Ventana**: 6 segundos (3 ciclos de frecuencia mínima)
- **Ventaja**: Balance entre velocidad y resolución temporal
- **Limitación**: Resolución fija (principio de incertidumbre)

### CWT (Continuous Wavelet Transform)
- **Propósito**: Análisis tiempo-frecuencia con resolución adaptativa
- **Wavelet**: Complex Morlet (cmor)
- **Escalas**: 50 escalas logarítmicas
- **Ventaja**: Resolución óptima para cada banda de frecuencia
- **Limitación**: Computacionalmente más costosa

## ⚡ Rendimiento

### Tiempos de Procesamiento (FileEEG.mat)
- **FFT**: 0.016 segundos
- **STFT**: 0.002 segundos  
- **CWT**: 1.138 segundos

### Comparación
- CWT es **70x más lento** que FFT
- CWT es **561x más lento** que STFT

## 📋 Conclusiones

### ¿Qué contenido en frecuencia identifica cada transformada?

1. **FFT**: Contenido frecuencial promedio de toda la señal
   - Identifica bandas dominantes (delta, theta, alpha, beta, gamma)
   - Sin información temporal

2. **STFT**: Contenido frecuencial con resolución temporal fija
   - Buena para eventos transitorios
   - Resolución limitada por principio de incertidumbre

3. **CWT**: Contenido frecuencial con resolución adaptativa
   - Resolución temporal alta para frecuencias altas
   - Resolución frecuencial alta para frecuencias bajas
   - Ideal para análisis simultáneo de múltiples bandas EEG

## 🌐 GitLab Pages

Este proyecto está configurado para desplegarse automáticamente en GitLab Pages:

1. Sube el código a GitLab
2. El pipeline CI/CD ejecutará el análisis automáticamente
3. El dashboard estará disponible en: `https://[usuario].gitlab.io/[proyecto]`

## 📁 Estructura del Proyecto

```
├── analysis.py              # Script principal de análisis
├── requirements.txt         # Dependencias Python
├── .gitlab-ci.yml          # Configuración CI/CD
├── .gitignore              # Archivos a ignorar
├── README.md               # Este archivo
├── FileEEG.mat            # Datos EEG (1024 Hz)
├── sEEG.mat               # Datos EEG (256 Hz)
├── dashboard.html         # Dashboard generado
└── analysis_report.md     # Reporte generado
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.10**
- **NumPy**: Manipulación de arrays
- **SciPy**: Transformadas y procesamiento de señales
- **PyWavelets**: Transformada Wavelet Continua
- **Plotly**: Visualizaciones interactivas
- **Matplotlib**: Gráficas estáticas

## 👨‍💻 Autor

**Felipe Rangel**  
Procesamiento de Señales Biológicas  
Maestría en Inteligencia Artificial  
Universidad Javeriana

## 📅 Fecha

Octubre 2025

---

*Este proyecto forma parte del curso de Procesamiento de Señales Biológicas y demuestra la implementación práctica de transformadas tiempo-frecuencia para análisis de señales EEG.*
