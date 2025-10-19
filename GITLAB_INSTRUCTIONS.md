# Instrucciones para Subir a GitLab

## 🚀 Pasos para Desplegar en GitLab Pages

### 1. Crear Repositorio en GitLab

1. Ve a [GitLab.com](https://gitlab.com) y crea una cuenta si no tienes una
2. Haz clic en "New Project" → "Create blank project"
3. Nombre del proyecto: `analisis-eeg-fft-stft-cwt` (o el que prefieras)
4. Descripción: "Análisis comparativo de señales EEG con FFT, STFT y CWT"
5. Visibilidad: Public (para GitLab Pages gratuito)
6. NO inicialices con README (ya tenemos uno)

### 2. Subir Código

```bash
# Agregar remote de GitLab
git remote add origin https://gitlab.com/[TU_USUARIO]/[NOMBRE_PROYECTO].git

# Subir código
git push -u origin main
```

### 3. Activar GitLab Pages

1. Ve a tu proyecto en GitLab
2. Navega a **Settings** → **Pages**
3. El pipeline CI/CD se ejecutará automáticamente
4. Una vez completado, tu dashboard estará disponible en:
   `https://[TU_USUARIO].gitlab.io/[NOMBRE_PROYECTO]`

### 4. Verificar Pipeline

1. Ve a **CI/CD** → **Pipelines**
2. Verifica que el pipeline se ejecute exitosamente
3. Si hay errores, revisa los logs y corrige

## 📋 Archivos Incluidos

- ✅ `analysis.py` - Script principal
- ✅ `dashboard.html` - Dashboard interactivo (135MB)
- ✅ `analysis_report.md` - Reporte detallado
- ✅ `requirements.txt` - Dependencias Python
- ✅ `.gitlab-ci.yml` - Configuración CI/CD
- ✅ `README.md` - Documentación completa
- ✅ `FileEEG.mat` - Datos EEG (1024 Hz)
- ✅ `sEEG.mat` - Datos EEG (256 Hz)

## 🔧 Configuración del Pipeline

El archivo `.gitlab-ci.yml` está configurado para:

1. **Stage 1 - Analyze**: Ejecutar el análisis Python
2. **Stage 2 - Deploy**: Desplegar a GitLab Pages

### Variables del Pipeline

- **Python Version**: 3.10
- **Dependencies**: Instaladas automáticamente desde `requirements.txt`
- **Artifacts**: `dashboard.html` y `analysis_report.md`
- **Pages**: Desplegadas automáticamente

## 🌐 URL Final

Una vez desplegado, tu dashboard estará disponible en:
```
https://[TU_USUARIO].gitlab.io/[NOMBRE_PROYECTO]
```

## 📊 Contenido del Dashboard

El dashboard incluye:

1. **FileEEG**: 
   - FFT: Espectro con picos en 0.42, 0.49, 0.56, 0.62, 0.70 Hz
   - STFT: Espectrograma con ventana de 6 segundos
   - CWT: Escalograma con 50 escalas

2. **sEEG**:
   - FFT: Espectro con picos en 0.35, 0.37, 0.40, 0.42, 0.46 Hz
   - STFT: Espectrograma con ventana de 6 segundos
   - CWT: Escalograma con 50 escalas

## ⚡ Rendimiento Observado

- **FFT**: ~0.01 segundos (más rápida)
- **STFT**: ~0.002 segundos (más rápida aún)
- **CWT**: ~1.1 segundos (más lenta pero mejor resolución)

## 🎯 Respuesta a la Pregunta Principal

**¿Qué contenido en frecuencia identifica cada transformada?**

1. **FFT**: Contenido frecuencial promedio de toda la señal
2. **STFT**: Contenido frecuencial con resolución temporal fija
3. **CWT**: Contenido frecuencial con resolución temporal adaptativa

## 🆘 Solución de Problemas

### Pipeline Falla
- Verifica que `requirements.txt` tenga todas las dependencias
- Revisa los logs del pipeline en GitLab
- Asegúrate de que los archivos `.mat` estén incluidos

### Pages No Se Despliegan
- Verifica que el pipeline haya completado exitosamente
- Revisa la configuración en Settings → Pages
- Espera unos minutos para la propagación DNS

### Dashboard No Carga
- Verifica que `dashboard.html` se haya generado correctamente
- Revisa el tamaño del archivo (debe ser ~135MB)
- Prueba abrir el archivo localmente primero

---

¡Tu análisis de señales EEG estará disponible públicamente en GitLab Pages! 🎉
