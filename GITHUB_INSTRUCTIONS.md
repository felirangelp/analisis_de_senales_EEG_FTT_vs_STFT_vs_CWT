# 🚀 Instrucciones para Desplegar en GitHub Pages

## 📋 Pasos para Configurar GitHub Pages

### 1. Crear Repositorio en GitHub
1. Ve a [GitHub.com](https://github.com) y crea una cuenta si no tienes una
2. Haz clic en "New repository" (botón verde)
3. Nombre del repositorio: `eeg-analysis-dashboard` (o el nombre que prefieras)
4. Descripción: "Análisis comparativo de señales EEG: FFT vs STFT vs CWT"
5. Marca como **Público** (necesario para GitHub Pages gratuito)
6. **NO** inicialices con README, .gitignore o licencia (ya los tenemos)
7. Haz clic en "Create repository"

### 2. Conectar Repositorio Local con GitHub
```bash
# En tu terminal, desde la carpeta del proyecto:
git remote add origin https://github.com/TU-USUARIO/TU-REPOSITORIO.git
git branch -M main
git push -u origin main
```

**Reemplaza:**
- `TU-USUARIO`: Tu nombre de usuario de GitHub
- `TU-REPOSITORIO`: El nombre que le diste al repositorio

### 3. Configurar GitHub Pages
1. Ve a tu repositorio en GitHub
2. Haz clic en la pestaña **"Settings"** (arriba a la derecha)
3. Desplázate hacia abajo hasta la sección **"Pages"** (lado izquierdo)
4. En "Source", selecciona **"GitHub Actions"**
5. GitHub automáticamente detectará el workflow que creamos

### 4. Activar GitHub Actions
1. Ve a la pestaña **"Actions"** en tu repositorio
2. Deberías ver el workflow "Deploy EEG Analysis Dashboard to GitHub Pages"
3. Haz clic en él y luego en **"Run workflow"**
4. Selecciona la rama `main` y haz clic en **"Run workflow"**

### 5. Verificar Despliegue
1. Una vez que el workflow termine exitosamente (verde ✅)
2. Ve de nuevo a **Settings > Pages**
3. Verás la URL de tu sitio: `https://TU-USUARIO.github.io/TU-REPOSITORIO/`
4. Haz clic en la URL para ver tu dashboard

## 🔄 Flujo de Trabajo Automático

Cada vez que hagas `git push` a la rama `main`:
1. GitHub Actions ejecutará automáticamente el análisis
2. Generará los dashboards HTML
3. Los desplegará a GitHub Pages
4. Tu sitio se actualizará automáticamente

## 📁 Estructura Final del Sitio

Tu sitio web tendrá estas páginas:
- **Página Principal**: `https://TU-USUARIO.github.io/TU-REPOSITORIO/`
  - Dashboard con pestañas (gráficas + interpretación)
- **Solo Gráficas**: `https://TU-USUARIO.github.io/TU-REPOSITORIO/dashboard.html`
- **Solo Interpretación**: `https://TU-USUARIO.github.io/TU-REPOSITORIO/dashboard_interpretaciones.html`
- **Reporte**: `https://TU-USUARIO.github.io/TU-REPOSITORIO/analysis_report.md`

## 🛠️ Comandos Útiles

### Subir Cambios
```bash
git add .
git commit -m "Descripción de los cambios"
git push origin main
```

### Ver Estado del Repositorio
```bash
git status
git log --oneline
```

### Verificar URLs en README
Después de crear el repositorio, actualiza las URLs en `README.md`:
- Reemplaza `tu-usuario` con tu usuario de GitHub
- Reemplaza `tu-repositorio` con el nombre de tu repositorio

## 🎯 Características del Dashboard

### Dashboard Principal (index.html)
- **Navegación por pestañas**: Fácil cambio entre gráficas e interpretación
- **Diseño responsive**: Se adapta a móviles y tablets
- **Carga inteligente**: Solo carga el contenido de la pestaña activa
- **Interfaz moderna**: Gradientes, glassmorphism y animaciones

### Dashboard de Gráficas (dashboard.html)
- **Visualizaciones interactivas**: Zoom, pan, hover
- **Tres transformadas**: FFT, STFT, CWT
- **Dos señales**: FileEEG.mat y sEEG.mat
- **Comparación lado a lado**: Fácil análisis comparativo

### Dashboard de Interpretación (dashboard_interpretaciones.html)
- **Análisis detallado**: Características de cada transformada
- **Guía de colores**: Interpretación de mapas de colores
- **Métricas de rendimiento**: Tiempos de procesamiento
- **Conclusiones**: Recomendaciones de uso

## 🔧 Solución de Problemas

### El workflow no se ejecuta
- Verifica que el archivo `.github/workflows/deploy.yml` esté en el repositorio
- Asegúrate de que GitHub Actions esté habilitado en tu repositorio

### El sitio no se despliega
- Revisa los logs del workflow en la pestaña "Actions"
- Verifica que todos los archivos HTML se generen correctamente

### Las gráficas no cargan
- Verifica que `analysis.py` se ejecute sin errores
- Revisa que los archivos `.mat` estén en el repositorio

## 📞 Soporte

Si tienes problemas:
1. Revisa los logs de GitHub Actions
2. Verifica que todos los archivos estén en el repositorio
3. Asegúrate de que el repositorio sea público
4. Contacta al instructor si persisten los problemas

---

**¡Tu dashboard estará disponible en GitHub Pages en pocos minutos!** 🎉
