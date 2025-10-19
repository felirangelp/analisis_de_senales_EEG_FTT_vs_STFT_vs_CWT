#!/usr/bin/env python3
"""
Análisis de Señales EEG con FFT, STFT y CWT
============================================

Este script implementa tres transformadas para analizar señales EEG:
1. FFT (Fast Fourier Transform) - Espectro de frecuencias
2. STFT (Short-Time Fourier Transform) - Espectrograma
3. CWT (Continuous Wavelet Transform) - Escalograma

Autor: Felipe Rangel
Fecha: Octubre 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import pywt
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import os

class EEGAnalyzer:
    """Clase para análisis de señales EEG con múltiples transformadas."""
    
    def __init__(self):
        self.signals = {}
        self.fs = None  # Frecuencia de muestreo
        self.results = {}
        
    def load_data(self, file_path):
        """Cargar datos desde archivo .mat"""
        try:
            data = sio.loadmat(file_path)
            print(f"Archivo cargado: {file_path}")
            print(f"Variables disponibles: {list(data.keys())}")
            
            # Buscar la señal principal (excluyendo metadatos)
            signal_keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"Variables de señal: {signal_keys}")
            
            return data, signal_keys
        except Exception as e:
            print(f"Error cargando {file_path}: {e}")
            return None, []
    
    def explore_data(self, file_path):
        """Explorar estructura de los datos"""
        data, signal_keys = self.load_data(file_path)
        if data is None:
            return None
            
        print(f"\n=== Exploración de {file_path} ===")
        
        for key in signal_keys:
            signal_data = data[key]
            print(f"\nVariable: {key}")
            print(f"Tipo: {type(signal_data)}")
            print(f"Forma: {signal_data.shape}")
            print(f"Tipo de datos: {signal_data.dtype}")
            
            if signal_data.ndim == 1:
                print(f"Longitud: {len(signal_data)} muestras")
                if len(signal_data) > 0 and np.issubdtype(signal_data.dtype, np.number):
                    print(f"Rango: [{signal_data.min():.4f}, {signal_data.max():.4f}]")
                    print(f"Media: {signal_data.mean():.4f}")
                    print(f"Desv. std: {signal_data.std():.4f}")
                elif len(signal_data) > 0:
                    print(f"Tipo de datos no numérico: {signal_data.dtype}")
                    print(f"Primeros valores: {signal_data[:3]}")
            elif signal_data.ndim == 2:
                print(f"Dimensiones: {signal_data.shape[0]} filas x {signal_data.shape[1]} columnas")
                if np.issubdtype(signal_data.dtype, np.number):
                    print(f"Rango: [{signal_data.min():.4f}, {signal_data.max():.4f}]")
                else:
                    print(f"Tipo de datos no numérico: {signal_data.dtype}")
        
        return data, signal_keys
    
    def get_sampling_rate(self, data, signal_keys):
        """Obtener frecuencia de muestreo desde los datos"""
        # Buscar variable de frecuencia de muestreo
        fs_candidates = ['Fs', 'fs', 'sampling_rate', 'freq']
        
        for candidate in fs_candidates:
            if candidate in signal_keys:
                fs_value = data[candidate]
                if hasattr(fs_value, 'item'):
                    fs = int(fs_value.item())
                else:
                    fs = int(fs_value[0])
                print(f"Frecuencia de muestreo encontrada: {fs} Hz")
                return fs
        
        # Si no se encuentra, estimar
        print("Frecuencia de muestreo no encontrada, estimando...")
        return self.estimate_sampling_rate(data[signal_keys[0]])
    
    def estimate_sampling_rate(self, signal_data):
        """Estimar frecuencia de muestreo basada en la longitud de la señal"""
        # Asumir duración típica de EEG (ej: 3 minutos)
        fs_estimated = len(signal_data) / 180.0
        
        # Redondear a valores típicos de EEG
        common_fs = [250, 500, 1000, 1024, 2000]
        fs = min(common_fs, key=lambda x: abs(x - fs_estimated))
        
        print(f"Frecuencia de muestreo estimada: {fs} Hz")
        return fs
    
    def preprocess_signal(self, signal_data, fs):
        """Preprocesar señal EEG"""
        # Remover tendencia (detrend)
        signal_clean = signal.detrend(signal_data)
        
        # Solo aplicar filtro si la señal es suficientemente larga
        if len(signal_clean) > 1000:  # Mínimo 1000 muestras
            try:
                # Filtro pasa-banda típico para EEG (0.5-50 Hz)
                nyquist = fs / 2
                low = 0.5 / nyquist
                high = min(50.0 / nyquist, 0.99)  # Evitar problemas de Nyquist
                
                # Diseñar filtro Butterworth
                b, a = signal.butter(4, [low, high], btype='band')
                signal_filtered = signal.filtfilt(b, a, signal_clean)
                print(f"Filtro aplicado: {0.5}-{50} Hz")
            except Exception as e:
                print(f"Error en filtrado: {e}. Usando señal sin filtrar.")
                signal_filtered = signal_clean
        else:
            print("Señal muy corta para filtrado. Usando señal sin filtrar.")
            signal_filtered = signal_clean
        
        return signal_filtered
    
    def compute_fft(self, signal_data, fs):
        """Calcular FFT y obtener espectro de frecuencias"""
        print("Calculando FFT...")
        start_time = time.time()
        
        # Calcular FFT
        fft_result = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/fs)
        
        # Solo frecuencias positivas (como se mencionó en clase)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_result[:len(fft_result)//2])
        
        # Identificar picos principales
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(positive_fft, height=np.max(positive_fft)*0.1, distance=fs//100)
        
        processing_time = time.time() - start_time
        
        result = {
            'frequencies': positive_freqs,
            'magnitude': positive_fft,
            'peaks': peaks,
            'peak_freqs': positive_freqs[peaks],
            'peak_magnitudes': positive_fft[peaks],
            'processing_time': processing_time
        }
        
        print(f"FFT completado en {processing_time:.4f} segundos")
        if len(peaks) > 0:
            peak_freqs_str = ', '.join([f"{freq:.2f}" for freq in positive_freqs[peaks][:5]])
            print(f"Picos encontrados en frecuencias: {peak_freqs_str} Hz")
        else:
            print("No se encontraron picos significativos")
        
        return result
    
    def compute_stft(self, signal_data, fs, window_length=None):
        """Calcular STFT y generar espectrograma"""
        print("Calculando STFT...")
        start_time = time.time()
        
        # Determinar longitud de ventana (2-3 ciclos de frecuencia mínima de interés)
        if window_length is None:
            # Para EEG, frecuencia mínima típica es 0.5 Hz
            min_freq = 0.5
            cycles = 3
            window_length = int(cycles * fs / min_freq)
        
        # Asegurar que la ventana no sea más larga que la señal
        window_length = min(window_length, len(signal_data) // 4)
        
        # Calcular STFT
        f, t, Zxx = signal.stft(signal_data, fs, nperseg=window_length, 
                               noverlap=window_length//2, window='hann')
        
        processing_time = time.time() - start_time
        
        result = {
            'frequencies': f,
            'times': t,
            'magnitude': np.abs(Zxx),
            'phase': np.angle(Zxx),
            'window_length': window_length,
            'processing_time': processing_time
        }
        
        print(f"STFT completado en {processing_time:.4f} segundos")
        print(f"Ventana usada: {window_length} muestras ({window_length/fs:.2f} segundos)")
        
        return result
    
    def compute_cwt(self, signal_data, fs, wavelet='cmor', scales=None):
        """Calcular CWT y generar escalograma"""
        print("Calculando CWT...")
        start_time = time.time()
        
        # Definir escalas para bandas EEG típicas
        if scales is None:
            # Bandas EEG: delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-50)
            # Usar escalas logarítmicas para mejor resolución
            scales = np.logspace(0, 2, 50)  # De 1 a 100 escalas
        
        # Calcular CWT
        coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet, sampling_period=1/fs)
        
        processing_time = time.time() - start_time
        
        result = {
            'coefficients': coefficients,
            'scales': scales,
            'frequencies': frequencies,
            'magnitude': np.abs(coefficients),
            'phase': np.angle(coefficients),
            'wavelet': wavelet,
            'processing_time': processing_time
        }
        
        print(f"CWT completado en {processing_time:.4f} segundos")
        print(f"Escalas usadas: {len(scales)} (de {scales[0]:.2f} a {scales[-1]:.2f})")
        print(f"Rango de frecuencias: {frequencies[-1]:.2f} - {frequencies[0]:.2f} Hz")
        
        return result
    
    def analyze_signal(self, signal_name, signal_data, fs):
        """Realizar análisis completo de una señal"""
        print(f"\n=== Análisis completo de {signal_name} ===")
        
        results = {
            'signal_name': signal_name,
            'fs': fs,
            'duration': len(signal_data) / fs,
            'samples': len(signal_data)
        }
        
        # FFT
        results['fft'] = self.compute_fft(signal_data, fs)
        
        # STFT
        results['stft'] = self.compute_stft(signal_data, fs)
        
        # CWT
        results['cwt'] = self.compute_cwt(signal_data, fs)
        
        return results
    
    def create_dashboard(self):
        """Crear dashboard interactivo con Plotly"""
        print("\nGenerando dashboard interactivo...")
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=len(self.results), cols=3,
            subplot_titles=[f"{signal_name} - {transform}" 
                          for signal_name in self.results.keys() 
                          for transform in ['FFT', 'STFT', 'CWT']],
            specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(len(self.results))],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        row = 1
        for signal_name, results in self.results.items():
            fs = results['fs']
            
            # FFT - Espectro de frecuencias
            fft_data = results['fft']
            fig.add_trace(
                go.Scatter(
                    x=fft_data['frequencies'],
                    y=fft_data['magnitude'],
                    mode='lines',
                    name=f'{signal_name} FFT',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='Frecuencia: %{x:.2f} Hz<br>Magnitud: %{y:.2f}<extra></extra>'
                ),
                row=row, col=1
            )
            
            # Marcar picos principales
            if len(fft_data['peaks']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=fft_data['peak_freqs'][:5],
                        y=fft_data['peak_magnitudes'][:5],
                        mode='markers',
                        name=f'{signal_name} Picos',
                        marker=dict(color='red', size=8, symbol='diamond'),
                        hovertemplate='Pico: %{x:.2f} Hz<br>Magnitud: %{y:.2f}<extra></extra>'
                    ),
                    row=row, col=1
                )
            
            # STFT - Espectrograma
            stft_data = results['stft']
            fig.add_trace(
                go.Heatmap(
                    z=stft_data['magnitude'],
                    x=stft_data['times'],
                    y=stft_data['frequencies'],
                    colorscale='Viridis',
                    name=f'{signal_name} STFT',
                    hovertemplate='Tiempo: %{x:.2f}s<br>Frecuencia: %{y:.2f} Hz<br>Magnitud: %{z:.2f}<extra></extra>'
                ),
                row=row, col=2
            )
            
            # CWT - Escalograma
            cwt_data = results['cwt']
            fig.add_trace(
                go.Heatmap(
                    z=cwt_data['magnitude'],
                    x=np.arange(len(cwt_data['magnitude'][0])) / fs,
                    y=cwt_data['frequencies'],
                    colorscale='Plasma',
                    name=f'{signal_name} CWT',
                    hovertemplate='Tiempo: %{x:.2f}s<br>Frecuencia: %{y:.2f} Hz<br>Magnitud: %{z:.2f}<extra></extra>'
                ),
                row=row, col=3
            )
            
            row += 1
        
        # Actualizar layout
        fig.update_layout(
            title={
                'text': 'Análisis Comparativo de Señales EEG: FFT vs STFT vs CWT',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=400 * len(self.results),
            showlegend=False,
            template='plotly_dark',
            font=dict(family="Arial", size=12)
        )
        
        # Actualizar ejes
        for i in range(len(self.results)):
            # FFT
            fig.update_xaxes(title_text="Frecuencia (Hz)", row=i+1, col=1)
            fig.update_yaxes(title_text="Magnitud", row=i+1, col=1)
            
            # STFT
            fig.update_xaxes(title_text="Tiempo (s)", row=i+1, col=2)
            fig.update_yaxes(title_text="Frecuencia (Hz)", row=i+1, col=2)
            
            # CWT
            fig.update_xaxes(title_text="Tiempo (s)", row=i+1, col=3)
            fig.update_yaxes(title_text="Frecuencia (Hz)", row=i+1, col=3)
        
        # Guardar dashboard
        dashboard_path = 'dashboard.html'
        fig.write_html(dashboard_path)
        print(f"Dashboard guardado en: {dashboard_path}")
        
        return fig
    
    def generate_analysis_report(self):
        """Generar reporte de análisis"""
        print("\nGenerando reporte de análisis...")
        
        report = []
        report.append("# Análisis de Señales EEG: FFT, STFT y CWT")
        report.append("=" * 50)
        report.append("")
        
        for signal_name, results in self.results.items():
            report.append(f"## Señal: {signal_name}")
            report.append(f"- **Frecuencia de muestreo**: {results['fs']} Hz")
            report.append(f"- **Duración**: {results['duration']:.2f} segundos")
            report.append(f"- **Muestras**: {results['samples']:,}")
            report.append("")
            
            # FFT
            fft_data = results['fft']
            report.append("### FFT (Fast Fourier Transform)")
            report.append(f"- **Tiempo de procesamiento**: {fft_data['processing_time']:.4f} segundos")
            report.append("- **Contenido en frecuencia identificado**:")
            report.append("  - Identifica todas las componentes frecuenciales presentes en toda la señal")
            report.append("  - Sin resolución temporal (promedio de toda la señal)")
            report.append("  - Útil para identificar bandas de frecuencia dominantes")
            if len(fft_data['peaks']) > 0:
                peak_freqs = ', '.join([f"{freq:.2f}" for freq in fft_data['peak_freqs'][:5]])
                report.append(f"- **Picos principales**: {peak_freqs} Hz")
            report.append("")
            
            # STFT
            stft_data = results['stft']
            report.append("### STFT (Short-Time Fourier Transform)")
            report.append(f"- **Tiempo de procesamiento**: {stft_data['processing_time']:.4f} segundos")
            report.append(f"- **Ventana utilizada**: {stft_data['window_length']} muestras ({stft_data['window_length']/results['fs']:.2f} segundos)")
            report.append("- **Contenido en frecuencia identificado**:")
            report.append("  - Identifica contenido frecuencial con resolución temporal fija")
            report.append("  - Ventana fija: buena resolución temporal para frecuencias altas")
            report.append("  - Limitada resolución frecuencial para frecuencias bajas")
            report.append("  - Ideal para análisis de eventos transitorios")
            report.append("")
            
            # CWT
            cwt_data = results['cwt']
            report.append("### CWT (Continuous Wavelet Transform)")
            report.append(f"- **Tiempo de procesamiento**: {cwt_data['processing_time']:.4f} segundos")
            report.append(f"- **Wavelet utilizada**: {cwt_data['wavelet']}")
            report.append(f"- **Escalas**: {len(cwt_data['scales'])} (de {cwt_data['scales'][0]:.2f} a {cwt_data['scales'][-1]:.2f})")
            report.append("- **Contenido en frecuencia identificado**:")
            report.append("  - Identifica contenido frecuencial con resolución temporal adaptativa")
            report.append("  - Resolución temporal alta para frecuencias altas")
            report.append("  - Resolución frecuencial alta para frecuencias bajas")
            report.append("  - Mejor para análisis de diferentes bandas EEG simultáneamente")
            report.append("")
            
            # Comparación de tiempos
            fft_time = fft_data['processing_time']
            stft_time = stft_data['processing_time']
            cwt_time = cwt_data['processing_time']
            
            report.append("### Comparación de Rendimiento")
            report.append(f"- **CWT es {cwt_time/fft_time:.1f}x más lento que FFT**")
            report.append(f"- **CWT es {cwt_time/stft_time:.1f}x más lento que STFT**")
            report.append("")
            report.append("---")
            report.append("")
        
        # Conclusiones generales
        report.append("## Conclusiones Generales")
        report.append("")
        report.append("### ¿Qué contenido en frecuencia identifica cada transformada?")
        report.append("")
        report.append("1. **FFT**: Identifica el contenido frecuencial promedio de toda la señal")
        report.append("   - Ventaja: Muy rápida, buena para identificar bandas dominantes")
        report.append("   - Limitación: No proporciona información temporal")
        report.append("")
        report.append("2. **STFT**: Identifica contenido frecuencial con resolución temporal fija")
        report.append("   - Ventaja: Balance entre velocidad y resolución temporal")
        report.append("   - Limitación: Resolución fija (principio de incertidumbre)")
        report.append("")
        report.append("3. **CWT**: Identifica contenido frecuencial con resolución adaptativa")
        report.append("   - Ventaja: Resolución óptima para cada banda de frecuencia")
        report.append("   - Limitación: Computacionalmente más costosa")
        report.append("")
        report.append("### Recomendaciones de Uso")
        report.append("- **FFT**: Para análisis inicial y identificación de bandas dominantes")
        report.append("- **STFT**: Para análisis de eventos transitorios y tiempo real")
        report.append("- **CWT**: Para análisis detallado de múltiples bandas EEG simultáneamente")
        
        # Guardar reporte
        report_text = '\n'.join(report)
        with open('analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("Reporte guardado en: analysis_report.md")
        return report_text

def main():
    """Función principal para explorar los datos"""
    analyzer = EEGAnalyzer()
    
    # Archivos a analizar
    files = ['FileEEG.mat', 'sEEG.mat']
    
    print("=== ANÁLISIS DE SEÑALES EEG ===")
    print("Implementando FFT, STFT y CWT")
    print("=" * 50)
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f"\nProcesando: {file_path}")
            data, signal_keys = analyzer.explore_data(file_path)
            
            if data is not None and signal_keys:
                # Obtener frecuencia de muestreo
                fs = analyzer.get_sampling_rate(data, signal_keys)
                
                # Buscar la variable de señal (excluyendo Fs)
                signal_key = None
                for key in signal_keys:
                    if key not in ['Fs', 'fs', 'sampling_rate', 'freq']:
                        signal_key = key
                        break
                
                if signal_key is None:
                    print("No se encontró variable de señal")
                    continue
                    
                signal_data = data[signal_key]
                
                # Si es multidimensional, tomar el primer canal
                if signal_data.ndim > 1:
                    signal_data = signal_data[:, 0] if signal_data.shape[0] > signal_data.shape[1] else signal_data[0, :]
                
                # Preprocesar señal
                signal_processed = analyzer.preprocess_signal(signal_data, fs)
                
                # Guardar datos procesados
                analyzer.signals[file_path] = {
                    'raw': signal_data,
                    'processed': signal_processed,
                    'fs': fs,
                    'duration': len(signal_processed) / fs
                }
                
                print(f"Duración de la señal: {analyzer.signals[file_path]['duration']:.2f} segundos")
                print(f"Número de muestras: {len(signal_processed)}")
                
        else:
            print(f"Archivo no encontrado: {file_path}")
    
    # Realizar análisis completo de cada señal
    print("\n" + "="*60)
    print("INICIANDO ANÁLISIS COMPLETO CON FFT, STFT Y CWT")
    print("="*60)
    
    for file_path, signal_info in analyzer.signals.items():
        signal_name = file_path.replace('.mat', '')
        signal_data = signal_info['processed']
        fs = signal_info['fs']
        
        # Realizar análisis completo
        results = analyzer.analyze_signal(signal_name, signal_data, fs)
        analyzer.results[signal_name] = results
        
        # Mostrar resumen de tiempos
        print(f"\n--- Resumen de tiempos para {signal_name} ---")
        print(f"FFT: {results['fft']['processing_time']:.4f} segundos")
        print(f"STFT: {results['stft']['processing_time']:.4f} segundos")
        print(f"CWT: {results['cwt']['processing_time']:.4f} segundos")
        
        # Comparar tiempos
        fft_time = results['fft']['processing_time']
        stft_time = results['stft']['processing_time']
        cwt_time = results['cwt']['processing_time']
        
        print(f"CWT es {cwt_time/fft_time:.1f}x más lento que FFT")
        print(f"CWT es {cwt_time/stft_time:.1f}x más lento que STFT")
    
    # Generar dashboard y reporte
    print("\n" + "="*60)
    print("GENERANDO DASHBOARD Y REPORTE")
    print("="*60)
    
    # Crear dashboard interactivo
    fig = analyzer.create_dashboard()
    
    # Generar reporte de análisis
    report = analyzer.generate_analysis_report()
    
    print("\n¡Análisis completado exitosamente!")
    print("Archivos generados:")
    print("- dashboard.html (Dashboard interactivo)")
    print("- analysis_report.md (Reporte detallado)")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
