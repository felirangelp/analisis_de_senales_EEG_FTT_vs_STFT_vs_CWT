# 🧠 Análisis de Señales EEG: FFT vs STFT vs CWT

Este proyecto realiza un análisis comparativo de señales electroencefalográficas (EEG) utilizando tres transformadas diferentes:

- **FFT (Fast Fourier Transform)**: Análisis en el dominio de la frecuencia
- **STFT (Short-Time Fourier Transform)**: Análisis tiempo-frecuencia con ventanas fijas
- **CWT (Continuous Wavelet Transform)**: Análisis tiempo-frecuencia con ventanas adaptativas

## 📊 Dashboards Disponibles

### 🎯 [Dashboard Principal](https://tu-usuario.github.io/tu-repositorio/)
Página principal con pestañas que integra ambos dashboards:
- **Pestaña Gráficas**: Visualizaciones interactivas de las transformadas
- **Pestaña Interpretación**: Análisis detallado y comentarios interpretativos

### 📈 [Gráficas Interactivas](https://tu-usuario.github.io/tu-repositorio/dashboard.html)
Dashboard enfocado únicamente en las visualizaciones:
- Gráficas de FFT con picos identificados
- Espectrogramas STFT con mapa de colores viridis
- Escalogramas CWT con mapa de colores plasma

### 📝 [Interpretación Detallada](https://tu-usuario.github.io/tu-repositorio/dashboard_interpretaciones.html)
Dashboard con análisis interpretativo completo:
- Características de cada transformada
- Comparación de rendimiento computacional
- Guía de colores y significados
- Conclusiones y recomendaciones

## 🚀 Instalación y Uso

### Requisitos
- Python 3.10+
- pip

### Instalación
```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio

# Crear ambiente virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución
```bash
# Ejecutar análisis completo
python analysis.py
```

## 📁 Archivos del Proyecto

- `analysis.py`: Script principal de análisis
- `requirements.txt`: Dependencias de Python
- `FileEEG.mat` y `sEEG.mat`: Archivos de datos EEG
- `index.html`: Dashboard principal con pestañas
- `dashboard.html`: Dashboard de gráficas
- `dashboard_interpretaciones.html`: Dashboard de interpretación
- `analysis_report.md`: Reporte detallado del análisis

## 🔧 Tecnologías Utilizadas

- **Python**: Lenguaje principal
- **SciPy**: Procesamiento de señales y filtros
- **NumPy**: Operaciones numéricas
- **PyWavelets**: Transformada wavelet continua
- **Plotly**: Visualizaciones interactivas
- **GitHub Actions**: Despliegue automático
- **GitHub Pages**: Hosting del dashboard

## 📈 Características del Análisis

### Preprocesamiento
- Detrending de señales
- Filtro pasa-banda (0.5-50 Hz)
- Normalización de datos

### Transformadas Implementadas
1. **FFT**: Identificación de componentes frecuenciales principales
2. **STFT**: Análisis tiempo-frecuencia con ventana Hanning
3. **CWT**: Análisis multiresolución con wavelet Morlet compleja

### Métricas de Rendimiento
- Tiempo de procesamiento para cada transformada
- Comparación de eficiencia computacional
- Análisis de resolución temporal y frecuencial

## 🎨 Diseño del Dashboard

- **Interfaz moderna**: Diseño responsive con gradientes y efectos glassmorphism
- **Navegación por pestañas**: Acceso fácil a diferentes secciones
- **Gráficas interactivas**: Zoom, pan y hover con información detallada
- **Guía de colores**: Interpretación visual de los mapas de colores
- **Responsive**: Compatible con dispositivos móviles y tablets

## 📊 Resultados Destacados

- **FFT**: Identificación rápida de componentes frecuenciales dominantes
- **STFT**: Balance entre resolución temporal y frecuencial
- **CWT**: Mejor resolución para frecuencias variables en el tiempo

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Proyecto académico** - Procesamiento de Señales Biológicas  
Universidad Javeriana - Maestría en Inteligencia Artificial