# Entregables — HackMTY26

## Entregable 1 — Prototipo funcional

Sistema que toma una **grabación de conversación telefónica** y decide si el **caller entrante** es humano real o voz sintética.

## Entregable 2 — Endpoint HTTP `POST /detect`

Obligatorio para la ronda automatizada.

### Contrato

```
ENTRADA
  WAV estéreo, 8 kHz, base64-encoded
  channel 0 = caller   ← lo que hay que clasificar
  channel 1 = agente   ← contexto

SALIDA
  {
    "is_synthetic": true,   // REQUERIDO (bool)
    "confidence": 0.87      // opcional pero RECOMENDADO
  }
```

### Requisitos

- Endpoint **alcanzable durante toda la ventana de scoring** y durante el judging
- `confidence` se usa para **desempates** y para **premiar sistemas bien calibrados**
- Modelo, lenguaje, framework y hosting son **libres**. Solo importa que responda el contrato

## Entregable 3 — README corto

Explicando el enfoque usado.

## Entregable 4 — Defensa en vivo (15 minutos)

Los jueces visitan cada equipo. Durante esos 15 minutos:

- Corren **su benchmark contra tu endpoint** usando un **test set oculto**
- Tú les explicas la solución

> El endpoint tiene que seguir arriba mientras te evalúan. No es opcional.

---

## Criterios de scoring

| Criterio | Qué evalúan |
| ---------- | ------------- |
| **Robustness** | desempeño ante speakers, motores y condiciones no vistos |
| **Originality** | ¿usa señales más allá de un clasificador off-the-shelf? |
| **Technical depth** | qué tan bien ejecutado está y si entiendes por qué funciona |
| **Feasibility** | ¿un banco podría desplegarlo sobre audio telefónico real? |
| **Latency** | qué tan rápido se llega a un resultado con confianza razonable |

> Latencia es criterio explícito. Modelo enorme que tarda 30s por llamada penaliza.

---

## Checklist de entregables

- [ ] Dataset descargado y explorado (conteo, balance, duración, codecs)
- [ ] Split **speaker-disjoint** propio para validación honesta
- [ ] Baseline acústico con embeddings congelados corriendo end-to-end
- [ ] Features conversacionales extraídas usando **ambos canales**
- [ ] `POST /detect` implementado con contrato exacto (base64, estéreo, 8 kHz, ch0=caller)
- [ ] `confidence` **calibrado**, no arbitrario
- [ ] Medición de latencia por llamada documentada
- [ ] README corto del enfoque
- [ ] Deploy estable y alcanzable durante toda la ventana de scoring
- [ ] Slides de 15 min: problema → señal(es) usada(s) → por qué funciona → límites honestos
