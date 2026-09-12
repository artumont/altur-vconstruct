# Scope — HackMTY26

## El problema

La voz solía ser prueba de identidad. Ya no. Con unos segundos de audio público cualquiera clona una voz.

**Centros de contacto** son la puerta de entrada de todo banco y hoy son atacados desde ambas direcciones:

- Defraudadores que **suplantan al cliente** para tomar control de cuentas
- Defraudadores que **suplantan al banco** para ingeniería social

Los más expuestos son quienes bancan por teléfono. La mayoría de centros de contacto **no tienen forma sólida de saber si la voz al otro lado es real**.

## Restricciones del dataset

| Aspecto | Realidad | Consecuencia |
| --------- | ---------- | -------------- |
| Audio 16 kHz lab | **8 kHz telefónico real** | Detectores SSL a 16 kHz se degradan. Resamplear o adaptar |
| Dataset grande | **Cientos de llamadas** | No entrenar desde cero. Transfer learning |
| Solo sospechoso | **Ambos canales disponibles** | Turnos del agente son contexto utilizable |
| Clip limpio 3s | **Conversación completa** | Riesgo alto de trampa de ventanas intermedias |
| Métrica EER | **Veredicto + confidence** | Calibración importa tanto como accuracy |
| Datos externos restringidos | **PDF no prohíbe datos externos** | Se puede preentrenar fuera; verificar con Altur |

## Cadena de degradación telefónica

```
voz sintetizada → enhancement → encoding (G.711/Opus/AAC)
→ compresión → transmisión → decoding → reconstrucción → receptor
```

- Telefonía tradicional: **8 kHz narrowband**, codec **G.711 μ-law/A-law**
- Resultado: artefactos de spoofing se atúan o reforman

## Señales de detección

### Dirección A — Acústica

El voice en sí. Artefactos espectrales, prosodia, respiración, huellas de modelo.

### Dirección B — Comportamiento conversacional

Cómo el caller maneja el flujo de conversación real.

- Humanos se recuperan de interrupciones de forma **instantánea y desordenada**
- Las máquinas se recuperan de forma **consistente** — la consistencia es señal

**Features clave:**

- Latencia de respuesta del caller tras cada turno del agente
- Reacción a interrupciones y solapamiento
- Regularidad de latencias (consistencia = señal)
- Comportamiento en silencios largos

### Dirección C — Semántica

Qué dice el caller. El agente pregunta cosas que **no existen**.

- Humano: *«no tengo eso»*
- Modelo de lenguaje: **inventa una respuesta**

## Frase clave de Altur

> «Depth beats breadth. One signal, done well, will outscore three that half-work.»

## Riesgos específicos

1. **8 kHz + dataset chico** — peor escenario para detector SSL. Mitigación: embeddings congelados + augmentación agresiva
2. **Ventanas intermedias** — puntuar audio real puntuando en medio da 0.9+ fake. Definir política de agregación
3. **Latencia puntuada** — inferencia con SSL es lenta. Considerar tramos representativos, batching, ONNX INT8
4. **Calibración** — confidence rompe empates. Usar Platt / isotonic
5. **Desbalance y motores ocultos** — test set tiene speakers/motores no vistos. Entrenar con diversidad
6. **Endpoint vivo durante judging** — deploy estable es parte del entregable
