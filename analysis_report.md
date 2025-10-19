# Análisis de Señales EEG: FFT, STFT y CWT
==================================================

## Señal: FileEEG
- **Frecuencia de muestreo**: 1024 Hz
- **Duración**: 180.00 segundos
- **Muestras**: 184,321

### FFT (Fast Fourier Transform)
- **Tiempo de procesamiento**: 0.0172 segundos
- **Contenido en frecuencia identificado**:
  - Identifica todas las componentes frecuenciales presentes en toda la señal
  - Sin resolución temporal (promedio de toda la señal)
  - Útil para identificar bandas de frecuencia dominantes
- **Picos principales**: 0.42, 0.49, 0.56, 0.62, 0.70 Hz

### STFT (Short-Time Fourier Transform)
- **Tiempo de procesamiento**: 0.0026 segundos
- **Ventana utilizada**: 6144 muestras (6.00 segundos)
- **Contenido en frecuencia identificado**:
  - Identifica contenido frecuencial con resolución temporal fija
  - Ventana fija: buena resolución temporal para frecuencias altas
  - Limitada resolución frecuencial para frecuencias bajas
  - Ideal para análisis de eventos transitorios

### CWT (Continuous Wavelet Transform)
- **Tiempo de procesamiento**: 1.0887 segundos
- **Wavelet utilizada**: cmor
- **Escalas**: 50 (de 1.00 a 100.00)
- **Contenido en frecuencia identificado**:
  - Identifica contenido frecuencial con resolución temporal adaptativa
  - Resolución temporal alta para frecuencias altas
  - Resolución frecuencial alta para frecuencias bajas
  - Mejor para análisis de diferentes bandas EEG simultáneamente

### Comparación de Rendimiento
- **CWT es 63.4x más lento que FFT**
- **CWT es 424.4x más lento que STFT**

---

## Señal: sEEG
- **Frecuencia de muestreo**: 256 Hz
- **Duración**: 78.13 segundos
- **Muestras**: 20,001

### FFT (Fast Fourier Transform)
- **Tiempo de procesamiento**: 0.0006 segundos
- **Contenido en frecuencia identificado**:
  - Identifica todas las componentes frecuenciales presentes en toda la señal
  - Sin resolución temporal (promedio de toda la señal)
  - Útil para identificar bandas de frecuencia dominantes
- **Picos principales**: 0.35, 0.37, 0.40, 0.42, 0.46 Hz

### STFT (Short-Time Fourier Transform)
- **Tiempo de procesamiento**: 0.0003 segundos
- **Ventana utilizada**: 1536 muestras (6.00 segundos)
- **Contenido en frecuencia identificado**:
  - Identifica contenido frecuencial con resolución temporal fija
  - Ventana fija: buena resolución temporal para frecuencias altas
  - Limitada resolución frecuencial para frecuencias bajas
  - Ideal para análisis de eventos transitorios

### CWT (Continuous Wavelet Transform)
- **Tiempo de procesamiento**: 0.1208 segundos
- **Wavelet utilizada**: cmor
- **Escalas**: 50 (de 1.00 a 100.00)
- **Contenido en frecuencia identificado**:
  - Identifica contenido frecuencial con resolución temporal adaptativa
  - Resolución temporal alta para frecuencias altas
  - Resolución frecuencial alta para frecuencias bajas
  - Mejor para análisis de diferentes bandas EEG simultáneamente

### Comparación de Rendimiento
- **CWT es 211.2x más lento que FFT**
- **CWT es 358.3x más lento que STFT**

---

## Conclusiones Generales

### ¿Qué contenido en frecuencia identifica cada transformada?

1. **FFT**: Identifica el contenido frecuencial promedio de toda la señal
   - Ventaja: Muy rápida, buena para identificar bandas dominantes
   - Limitación: No proporciona información temporal

2. **STFT**: Identifica contenido frecuencial con resolución temporal fija
   - Ventaja: Balance entre velocidad y resolución temporal
   - Limitación: Resolución fija (principio de incertidumbre)

3. **CWT**: Identifica contenido frecuencial con resolución adaptativa
   - Ventaja: Resolución óptima para cada banda de frecuencia
   - Limitación: Computacionalmente más costosa

### Recomendaciones de Uso
- **FFT**: Para análisis inicial y identificación de bandas dominantes
- **STFT**: Para análisis de eventos transitorios y tiempo real
- **CWT**: Para análisis detallado de múltiples bandas EEG simultáneamente