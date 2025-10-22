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
            
            # STFT - Espectrograma (reducir resolución para archivo más pequeño)
            stft_data = results['stft']
            # Reducir resolución tomando cada 4to punto
            z_reduced = stft_data['magnitude'][::4, ::4]
            x_reduced = stft_data['times'][::4]
            y_reduced = stft_data['frequencies'][::4]
            
            fig.add_trace(
                go.Heatmap(
                    z=z_reduced,
                    x=x_reduced,
                    y=y_reduced,
                    colorscale='Viridis',
                    name=f'{signal_name} STFT',
                    hovertemplate='Tiempo: %{x:.2f}s<br>Frecuencia: %{y:.2f} Hz<br>Magnitud: %{z:.2f}<extra></extra>'
                ),
                row=row, col=2
            )
            
            # CWT - Escalograma (reducir resolución para archivo más pequeño)
            cwt_data = results['cwt']
            # Reducir resolución tomando cada 4to punto
            z_reduced = cwt_data['magnitude'][::4, ::4]
            x_reduced = np.arange(len(cwt_data['magnitude'][0]))[::4] / fs
            y_reduced = cwt_data['frequencies'][::4]
            
            fig.add_trace(
                go.Heatmap(
                    z=z_reduced,
                    x=x_reduced,
                    y=y_reduced,
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
                'text': 'Análisis Comparativo de Señales EEG: FFT vs STFT vs CWT<br><sub>¿Qué contenido en frecuencia identifica cada transformada?</sub>',
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
            
            # STFT - Escala logarítmica en Y (frecuencia)
            fig.update_xaxes(title_text="Tiempo (s)", row=i+1, col=2)
            fig.update_yaxes(title_text="Frecuencia (Hz)", row=i+1, col=2, type="log")
            
            # CWT - Escala logarítmica en Y (frecuencia)
            fig.update_xaxes(title_text="Tiempo (s)", row=i+1, col=3)
            fig.update_yaxes(title_text="Frecuencia (Hz)", row=i+1, col=3, type="log")
        
        # Sin anotaciones - solo gráficas limpias
        
        # Guardar dashboard principal
        dashboard_path = 'dashboard.html'
        fig.write_html(dashboard_path, include_plotlyjs='cdn')
        print(f"Dashboard principal guardado en: {dashboard_path}")
        
        # Crear dashboard de interpretaciones
        self.create_interpretation_dashboard()
        
        return fig
    
    def create_interpretation_dashboard(self):
        """Crear dashboard separado con interpretaciones detalladas y diseño moderno"""
        print("\nGenerando dashboard de interpretaciones con diseño moderno...")
        
        # Crear HTML personalizado con diseño moderno
        html_content = self.create_modern_interpretation_html()
        
        # Guardar dashboard de interpretaciones
        interpretation_path = 'dashboard_interpretaciones.html'
        with open(interpretation_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Dashboard de interpretaciones moderno guardado en: {interpretation_path}")
        
        return None
    
    def create_modern_interpretation_html(self):
        """Crear HTML moderno para el dashboard de interpretaciones"""
        
        # Generar contenido interpretativo
        interpretation_data = self.generate_interpretation_data()
        
        html_template = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análisis Interpretativo: FFT vs STFT vs CWT</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .main-question {{
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        }}
        
        .main-question h2 {{
            font-size: 1.5em;
            margin-bottom: 10px;
        }}
        
        .main-question p {{
            font-size: 1.1em;
            font-weight: 300;
        }}
        
        .signal-analysis {{
            margin-bottom: 40px;
        }}
        
        .signal-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }}
        
        .signal-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .signal-icon {{
            width: 40px;
            height: 40px;
            background: linear-gradient(45deg, #3498db, #2980b9);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2em;
        }}
        
        .signal-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        
        .info-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }}
        
        .info-item strong {{
            color: #2c3e50;
        }}
        
        .transform-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 25px;
        }}
        
        .transform-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-top: 5px solid;
        }}
        
        .transform-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        }}
        
        .transform-card.fft {{
            border-top-color: #e74c3c;
        }}
        
        .transform-card.stft {{
            border-top-color: #27ae60;
        }}
        
        .transform-card.cwt {{
            border-top-color: #8e44ad;
        }}
        
        .transform-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .transform-icon {{
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .transform-icon.fft {{
            background: linear-gradient(45deg, #e74c3c, #c0392b);
        }}
        
        .transform-icon.stft {{
            background: linear-gradient(45deg, #27ae60, #229954);
        }}
        
        .transform-icon.cwt {{
            background: linear-gradient(45deg, #8e44ad, #7d3c98);
        }}
        
        .transform-title {{
            font-size: 1.4em;
            color: #2c3e50;
            font-weight: 600;
        }}
        
        .transform-content {{
            color: #555;
        }}
        
        .transform-content ul {{
            list-style: none;
            padding: 0;
        }}
        
        .transform-content li {{
            margin-bottom: 8px;
            padding-left: 20px;
            position: relative;
        }}
        
        .transform-content li:before {{
            content: "▶";
            position: absolute;
            left: 0;
            color: #3498db;
            font-size: 0.8em;
        }}
        
        .transform-content strong {{
            color: #2c3e50;
        }}
        
        .performance-section {{
            background: linear-gradient(45deg, #9b59b6, #8e44ad);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
            box-shadow: 0 8px 25px rgba(155, 89, 182, 0.3);
        }}
        
        .performance-title {{
            font-size: 1.5em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .performance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        
        .performance-item {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        
        .conclusions-section {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-top: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }}
        
        .conclusions-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 25px;
            text-align: center;
        }}
        
        .conclusion-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
        }}
        
        .conclusion-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            border-left: 5px solid;
        }}
        
        .conclusion-card:nth-child(1) {{
            border-left-color: #e74c3c;
        }}
        
        .conclusion-card:nth-child(2) {{
            border-left-color: #27ae60;
        }}
        
        .conclusion-card:nth-child(3) {{
            border-left-color: #8e44ad;
        }}
        
        .conclusion-title {{
            font-size: 1.3em;
            color: #2c3e50;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .recommendations {{
            background: linear-gradient(45deg, #1abc9c, #16a085);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-top: 30px;
            box-shadow: 0 8px 25px rgba(26, 188, 156, 0.3);
        }}
        
        .recommendations-title {{
            font-size: 1.5em;
            margin-bottom: 15px;
        }}
        
        .recommendations-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .recommendation-item {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        
        .color-guide {{
            background: #e74c3c;
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-top: 30px;
        }}
        
        .color-guide-title {{
            font-size: 1.5em;
            margin-bottom: 15px;
        }}
        
        .color-items {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .color-item {{
            background: #f0625a;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            color: white;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .transform-grid {{
                grid-template-columns: 1fr;
            }}
            
            .signal-info {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Análisis Interpretativo de EEG</h1>
            <p class="subtitle">FFT vs STFT vs CWT - Comparación de Transformadas</p>
        </div>
        
        <div class="main-question">
            <h2>🎯 Pregunta Principal</h2>
            <p>¿Qué contenido en frecuencia identifica cada transformada?</p>
        </div>
        
        {interpretation_data['signals_html']}
        
        <div class="performance-section">
            <h2 class="performance-title">⚡ Comparación de Rendimiento</h2>
            <div class="performance-grid">
                {interpretation_data['performance_html']}
            </div>
        </div>
        
        <div class="conclusions-section">
            <h2 class="conclusions-title">🎓 Conclusiones Generales</h2>
            <div class="conclusion-grid">
                {interpretation_data['conclusions_html']}
            </div>
        </div>
        
        <div class="recommendations">
            <h2 class="recommendations-title">📋 Recomendaciones de Uso</h2>
            <div class="recommendations-grid">
                {interpretation_data['recommendations_html']}
            </div>
        </div>
        
        <div class="color-guide">
            <h2 class="color-guide-title">🔍 Guía de Colores</h2>
            <div class="color-items">
                {interpretation_data['color_guide_html']}
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def generate_interpretation_data(self):
        """Generar datos estructurados para el HTML moderno"""
        data = {
            'signals_html': '',
            'performance_html': '',
            'conclusions_html': '',
            'recommendations_html': '',
            'color_guide_html': ''
        }
        
        # Generar HTML para cada señal
        for signal_name, results in self.results.items():
            signal_html = f"""
            <div class="signal-analysis">
                <div class="signal-card">
                    <h2 class="signal-title">
                        <div class="signal-icon">📊</div>
                        Análisis de {signal_name.upper()}
                    </h2>
                    
                    <div class="signal-info">
                        <div class="info-item">
                            <strong>Frecuencia de muestreo:</strong><br>
                            {results['fs']} Hz
                        </div>
                        <div class="info-item">
                            <strong>Duración:</strong><br>
                            {results['duration']:.1f} segundos
                        </div>
                        <div class="info-item">
                            <strong>Muestras:</strong><br>
                            {results['samples']:,}
                        </div>
                    </div>
                    
                    <div class="transform-grid">
                        {self.generate_transform_cards(signal_name, results)}
                    </div>
                </div>
            </div>
            """
            data['signals_html'] += signal_html
        
        # Generar HTML de rendimiento
        for signal_name, results in self.results.items():
            fft_time = results['fft']['processing_time']
            stft_time = results['stft']['processing_time']
            cwt_time = results['cwt']['processing_time']
            
            data['performance_html'] += f"""
            <div class="performance-item">
                <strong>{signal_name}:</strong><br>
                CWT es {cwt_time/fft_time:.1f}x más lenta que FFT<br>
                CWT es {cwt_time/stft_time:.1f}x más lenta que STFT
            </div>
            """
        
        # Generar HTML de conclusiones
        conclusions = [
            {
                'title': 'FFT - Análisis Global',
                'content': 'Identifica el contenido frecuencial promedio de toda la señal. Sin información temporal. Útil para identificar bandas dominantes (delta, theta, alpha, beta, gamma).'
            },
            {
                'title': 'STFT - Análisis Temporal Fijo',
                'content': 'Identifica contenido frecuencial con resolución temporal fija. Ventana fija: buena resolución temporal para frecuencias altas. Ideal para análisis de eventos transitorios.'
            },
            {
                'title': 'CWT - Análisis Temporal Adaptativo',
                'content': 'Identifica contenido frecuencial con resolución temporal adaptativa. Resolución temporal alta para frecuencias altas. Mejor para análisis de diferentes bandas EEG simultáneamente.'
            }
        ]
        
        for conclusion in conclusions:
            data['conclusions_html'] += f"""
            <div class="conclusion-card">
                <h3 class="conclusion-title">{conclusion['title']}</h3>
                <p>{conclusion['content']}</p>
            </div>
            """
        
        # Generar HTML de recomendaciones
        recommendations = [
            {'transform': 'FFT', 'use': 'Para análisis inicial y identificación de bandas dominantes'},
            {'transform': 'STFT', 'use': 'Para análisis de eventos transitorios y tiempo real'},
            {'transform': 'CWT', 'use': 'Para análisis detallado de múltiples bandas EEG simultáneamente'}
        ]
        
        for rec in recommendations:
            data['recommendations_html'] += f"""
            <div class="recommendation-item">
                <strong>{rec['transform']}:</strong><br>
                {rec['use']}
            </div>
            """
        
        # Generar HTML de guía de colores
        color_guide = [
            {'color': 'Azul oscuro', 'meaning': 'Baja energía/magnitud'},
            {'color': 'Amarillo', 'meaning': 'Alta energía/magnitud'},
            {'color': 'Puntos rojos', 'meaning': 'Picos identificados en FFT'},
            {'color': 'STFT (Viridis)', 'meaning': 'Verde = baja energía, Amarillo = alta energía'},
            {'color': 'CWT (Plasma)', 'meaning': 'Púrpura = baja energía, Amarillo = alta energía'}
        ]
        
        for color in color_guide:
            data['color_guide_html'] += f"""
            <div class="color-item">
                <strong>{color['color']}:</strong><br>
                {color['meaning']}
            </div>
            """
        
        return data
    
    def generate_transform_cards(self, signal_name, results):
        """Generar tarjetas de transformadas para una señal"""
        cards_html = ""
        
        transforms = [
            {
                'name': 'FFT',
                'class': 'fft',
                'icon': 'F',
                'data': results['fft'],
                'description': 'Fast Fourier Transform'
            },
            {
                'name': 'STFT',
                'class': 'stft',
                'icon': 'S',
                'data': results['stft'],
                'description': 'Short-Time Fourier Transform'
            },
            {
                'name': 'CWT',
                'class': 'cwt',
                'icon': 'C',
                'data': results['cwt'],
                'description': 'Continuous Wavelet Transform'
            }
        ]
        
        for transform in transforms:
            data = transform['data']
            
            # Contenido específico para cada transformada
            if transform['name'] == 'FFT':
                content = f"""
                <ul>
                    <li><strong>Identifica:</strong> Contenido frecuencial promedio de toda la señal</li>
                    <li><strong>Resolución temporal:</strong> Ninguna (promedio global)</li>
                    <li><strong>Ventaja:</strong> Muy rápida, ideal para identificar bandas dominantes</li>
                    <li><strong>Limitación:</strong> No proporciona información temporal</li>
                    <li><strong>Picos principales:</strong> {', '.join([f"{freq:.2f}" for freq in data['peak_freqs'][:5]])} Hz</li>
                    <li><strong>Tiempo de procesamiento:</strong> {data['processing_time']:.4f} segundos</li>
                </ul>
                """
            elif transform['name'] == 'STFT':
                content = f"""
                <ul>
                    <li><strong>Identifica:</strong> Contenido frecuencial con resolución temporal fija</li>
                    <li><strong>Resolución temporal:</strong> Fija (ventana constante)</li>
                    <li><strong>Ventana utilizada:</strong> {data['window_length']/results['fs']:.1f} segundos</li>
                    <li><strong>Ventaja:</strong> Balance entre velocidad y resolución temporal</li>
                    <li><strong>Limitación:</strong> Resolución fija (principio de incertidumbre)</li>
                    <li><strong>Ideal para:</strong> Análisis de eventos transitorios</li>
                    <li><strong>Tiempo de procesamiento:</strong> {data['processing_time']:.4f} segundos</li>
                </ul>
                """
            else:  # CWT
                content = f"""
                <ul>
                    <li><strong>Identifica:</strong> Contenido frecuencial con resolución temporal adaptativa</li>
                    <li><strong>Resolución temporal:</strong> Adaptativa (cambia con la frecuencia)</li>
                    <li><strong>Wavelet utilizada:</strong> {data['wavelet']}</li>
                    <li><strong>Escalas:</strong> {len(data['scales'])} (de {data['scales'][0]:.1f} a {data['scales'][-1]:.1f})</li>
                    <li><strong>Ventaja:</strong> Resolución óptima para cada banda de frecuencia</li>
                    <li><strong>Limitación:</strong> Computacionalmente más costosa</li>
                    <li><strong>Ideal para:</strong> Análisis simultáneo de múltiples bandas EEG</li>
                    <li><strong>Tiempo de procesamiento:</strong> {data['processing_time']:.4f} segundos</li>
                </ul>
                """
            
            cards_html += f"""
            <div class="transform-card {transform['class']}">
                <div class="transform-header">
                    <div class="transform-icon {transform['class']}">{transform['icon']}</div>
                    <h3 class="transform-title">{transform['name']}</h3>
                </div>
                <div class="transform-content">
                    {content}
                </div>
            </div>
            """
        
        return cards_html
    
    def generate_detailed_interpretation(self):
        """Generar contenido interpretativo detallado"""
        content = []
        
        # Resumen general
        content.append("<b>🎯 PREGUNTA PRINCIPAL:</b><br>")
        content.append("<i>¿Qué contenido en frecuencia identifica cada transformada?</i><br><br>")
        
        # Para cada señal
        for signal_name, results in self.results.items():
            content.append(f"<b>📈 ANÁLISIS DE {signal_name.upper()}</b><br>")
            content.append(f"• Frecuencia de muestreo: {results['fs']} Hz<br>")
            content.append(f"• Duración: {results['duration']:.1f} segundos<br>")
            content.append(f"• Muestras: {results['samples']:,}<br><br>")
            
            # FFT
            fft_data = results['fft']
            content.append("<b>🔵 FFT (Fast Fourier Transform):</b><br>")
            content.append("• <b>Identifica:</b> Contenido frecuencial promedio de toda la señal<br>")
            content.append("• <b>Resolución temporal:</b> Ninguna (promedio global)<br>")
            content.append("• <b>Ventaja:</b> Muy rápida, ideal para identificar bandas dominantes<br>")
            content.append("• <b>Limitación:</b> No proporciona información temporal<br>")
            if len(fft_data['peaks']) > 0:
                peak_freqs = ', '.join([f"{freq:.2f}" for freq in fft_data['peak_freqs'][:5]])
                content.append(f"• <b>Picos principales:</b> {peak_freqs} Hz<br>")
            content.append(f"• <b>Tiempo de procesamiento:</b> {fft_data['processing_time']:.4f} segundos<br><br>")
            
            # STFT
            stft_data = results['stft']
            content.append("<b>🟢 STFT (Short-Time Fourier Transform):</b><br>")
            content.append("• <b>Identifica:</b> Contenido frecuencial con resolución temporal fija<br>")
            content.append("• <b>Resolución temporal:</b> Fija (ventana constante)<br>")
            content.append("• <b>Ventana utilizada:</b> {:.1f} segundos<br>".format(stft_data['window_length']/results['fs']))
            content.append("• <b>Ventaja:</b> Balance entre velocidad y resolución temporal<br>")
            content.append("• <b>Limitación:</b> Resolución fija (principio de incertidumbre)<br>")
            content.append("• <b>Ideal para:</b> Análisis de eventos transitorios<br>")
            content.append(f"• <b>Tiempo de procesamiento:</b> {stft_data['processing_time']:.4f} segundos<br><br>")
            
            # CWT
            cwt_data = results['cwt']
            content.append("<b>🟡 CWT (Continuous Wavelet Transform):</b><br>")
            content.append("• <b>Identifica:</b> Contenido frecuencial con resolución temporal adaptativa<br>")
            content.append("• <b>Resolución temporal:</b> Adaptativa (cambia con la frecuencia)<br>")
            content.append(f"• <b>Wavelet utilizada:</b> {cwt_data['wavelet']}<br>")
            content.append(f"• <b>Escalas:</b> {len(cwt_data['scales'])} (de {cwt_data['scales'][0]:.1f} a {cwt_data['scales'][-1]:.1f})<br>")
            content.append("• <b>Ventaja:</b> Resolución óptima para cada banda de frecuencia<br>")
            content.append("• <b>Limitación:</b> Computacionalmente más costosa<br>")
            content.append("• <b>Ideal para:</b> Análisis simultáneo de múltiples bandas EEG<br>")
            content.append(f"• <b>Tiempo de procesamiento:</b> {cwt_data['processing_time']:.4f} segundos<br><br>")
            
            # Comparación de rendimiento
            fft_time = fft_data['processing_time']
            stft_time = stft_data['processing_time']
            cwt_time = cwt_data['processing_time']
            
            content.append("<b>⚡ COMPARACIÓN DE RENDIMIENTO:</b><br>")
            content.append(f"• CWT es {cwt_time/fft_time:.1f}x más lento que FFT<br>")
            content.append(f"• CWT es {cwt_time/stft_time:.1f}x más lento que STFT<br>")
            content.append(f"• STFT es {stft_time/fft_time:.1f}x más lento que FFT<br><br>")
            
            content.append("─" * 50 + "<br><br>")
        
        # Conclusiones generales
        content.append("<b>🎓 CONCLUSIONES GENERALES</b><br><br>")
        
        content.append("<b>1. FFT - Análisis Global:</b><br>")
        content.append("• Identifica el contenido frecuencial promedio de toda la señal<br>")
        content.append("• Sin información temporal<br>")
        content.append("• Útil para identificar bandas dominantes (delta, theta, alpha, beta, gamma)<br><br>")
        
        content.append("<b>2. STFT - Análisis Temporal Fijo:</b><br>")
        content.append("• Identifica contenido frecuencial con resolución temporal fija<br>")
        content.append("• Ventana fija: buena resolución temporal para frecuencias altas<br>")
        content.append("• Limitada resolución frecuencial para frecuencias bajas<br>")
        content.append("• Ideal para análisis de eventos transitorios<br><br>")
        
        content.append("<b>3. CWT - Análisis Temporal Adaptativo:</b><br>")
        content.append("• Identifica contenido frecuencial con resolución temporal adaptativa<br>")
        content.append("• Resolución temporal alta para frecuencias altas<br>")
        content.append("• Resolución frecuencial alta para frecuencias bajas<br>")
        content.append("• Mejor para análisis de diferentes bandas EEG simultáneamente<br><br>")
        
        content.append("<b>📋 RECOMENDACIONES DE USO:</b><br>")
        content.append("• <b>FFT:</b> Para análisis inicial y identificación de bandas dominantes<br>")
        content.append("• <b>STFT:</b> Para análisis de eventos transitorios y tiempo real<br>")
        content.append("• <b>CWT:</b> Para análisis detallado de múltiples bandas EEG simultáneamente<br><br>")
        
        content.append("<b>🔍 INTERPRETACIÓN DE COLORES:</b><br>")
        content.append("• <b>Azul oscuro:</b> Baja energía/magnitud<br>")
        content.append("• <b>Amarillo:</b> Alta energía/magnitud<br>")
        content.append("• <b>Puntos rojos:</b> Picos identificados en FFT<br>")
        content.append("• <b>STFT (Viridis):</b> Verde = baja energía, Amarillo = alta energía<br>")
        content.append("• <b>CWT (Plasma):</b> Púrpura = baja energía, Amarillo = alta energía<br>")
        
        return ''.join(content)
    
    def add_interpretation_annotations(self, fig):
        """Agregar anotaciones interpretativas al dashboard"""
        
        # Agregar información interpretativa en el lado derecho
        interpretation_text = self.generate_interpretation_summary()
        
        fig.add_annotation(
            x=1.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=interpretation_text,
            showarrow=False,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            font=dict(color="white", size=11),
            align="left",
            width=300
        )
        
        # Agregar información de rendimiento
        performance_text = "<b>⚡ RENDIMIENTO:</b><br>"
        for signal_name, results in self.results.items():
            fft_time = results['fft']['processing_time']
            stft_time = results['stft']['processing_time']
            cwt_time = results['cwt']['processing_time']
            performance_text += f"• {signal_name}: CWT {cwt_time/fft_time:.0f}x más lento que FFT<br>"
        
        fig.add_annotation(
            x=1.02,
            y=0.3,
            xref="paper",
            yref="paper",
            text=performance_text,
            showarrow=False,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            font=dict(color="white", size=11),
            align="left",
            width=300
        )
    
    def generate_interpretation_summary(self):
        """Generar resumen interpretativo para las anotaciones"""
        content = []
        
        content.append("<b>🎯 INTERPRETACIÓN:</b><br><br>")
        
        content.append("<b>🔵 FFT:</b><br>")
        content.append("• Contenido frecuencial promedio<br>")
        content.append("• Sin resolución temporal<br>")
        content.append("• Identifica bandas dominantes<br><br>")
        
        content.append("<b>🟢 STFT:</b><br>")
        content.append("• Resolución temporal fija<br>")
        content.append("• Ideal para eventos transitorios<br>")
        content.append("• Balance velocidad/resolución<br><br>")
        
        content.append("<b>🟡 CWT:</b><br>")
        content.append("• Resolución adaptativa<br>")
        content.append("• Mejor para múltiples bandas<br>")
        content.append("• Computacionalmente costosa<br><br>")
        
        content.append("<b>🔍 COLORES:</b><br>")
        content.append("• Azul oscuro: Baja energía<br>")
        content.append("• Amarillo: Alta energía<br>")
        content.append("• Puntos rojos: Picos FFT")
        
        return ''.join(content)
    
    
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
