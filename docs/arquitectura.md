# Arquitectura — HackMTY26

## Ruta recomendada (menor riesgo)

### Arquitectura en dos capas, con conversación completa como entrada

1. **Capa acústica (base, obligatoria)** — Front-end preentrenado + cabecera ligera
2. **Capa conversacional (diferenciador)** — Features de ambos canales
3. **Capa semántica (bonus)** — Originalidad

---

## Capa Acústica

### Opción principal: SSL + clasificador

- **Backbone:** `facebook/wav2vec2-xls-r-300m` o `microsoft/wavlm-large` como extractor congelado
- **Head:** Cabecera tipo AASIST encima
- **Entrenamiento:** Solo cabecera, embeddings congelados
- **Ventaja:** Ruta más viable con dataset chico

### Métricas de referencia

| Backbone | EER (ASVspoof 2021 LA) | Nota |
| ---------- | ------------------------ | ------ |
| **WavLM-large** | **2.29%** | Mejor medido |
| **XLS-R-300M + AASIST** | **2.65 – 2.84%** | Desplegado en VoiceGuard |
| **Wav2Vec2-large** | 3.09% | Baseline sólida |

### Modelos preentrenados

- [nii-yamagishilab/xls-r-2b-anti-deepfake](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake)
- [nii-yamagishilab/wav2vec-large-anti-deepfake](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake)
- [SpeechAntiSpoofingBenchmarks/AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST)

### Alternativas rápidas

- **XGBoost sobre LFCC/MFCC** — F1 = 0.95 reportado, entrena en minutos
- **AASIST sobre espectrograma** — entrenable desde cero y rápido
- **DSFNetTiny INT8** — 0.62 MB, ~30 ms inferencia CPU, EER 8.47%

---

## Capa Conversacional

Derivar features del diálogo usando **ambos canales**:

| Feature | Qué mide |
| --------- | ---------- |
| Latencia de respuesta | Tiempo caller después de turno agente |
| Reacción a interrupciones | Cómo maneja solapamiento |
| Regularidad de latencias | Consistencia (señal clave Altur) |
| Comportamiento en silencio | Qué hace cuando agente calla |

**Regla:** humanos se recuperan de forma **instantánea y desordenada**. Máquinas de forma **consistente**.

---

## Capa Semántica (bonus)

Usar preguntas de verificación del agente como test:

- Humano: *«no tengo eso»*
- Modelo de lenguaje: **inventa una respuesta**

---

## Pipeline de inferencia

```
Audio 8kHz WAV stereo
    ↓
Separar canales (ch0=caller, ch1=agente)
    ↓
Resample a 16kHz (para SSL)
    ↓
Extraer embeddings (wav2vec2/WavLM congelado)
    ↓
Clasificador (head AASIST) → score acústico
    ↓
Extraer features conversacionales (turnos, tiempos)
    ↓
Combinar señales → veredicto final + confidence
```

---

## Augmentación recomendada (RawBoost + canal)

- Convolución con respuestas de impulsos
- Ruido aditivo + coloreado
- **Simulación codec:** resample 8 kHz → G.711 μ-law → resample 16 kHz
- **Silencio inicial aleatorio (0–1 s)** — obligatorio
- Gain / clip / dropout de banda

---

## Estrategia de scoring para demo

Re-evaluar **prefijo creciente desde inicio**, cada 2s, cap a 15s:

- Primera respuesta: ~3s
- Veredicto final: 15s

---

## Referencia: VoiceGuard

| Benchmark | Resultado |
| ----------- | ----------- |
| ASVspoof 2021 LA | **2.84% EER** |
| Clones Kokoro / XTTS v2 | 100% / 100% |
| IndexTTS-2 | 97% |
| ElevenLabs-v3 (motor no visto) | 95.8% |
| Modelo edge INT8 | 0.62 MB, ~30 ms |
