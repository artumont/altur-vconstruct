# Datasets — HackMTY26

## Dataset de Altur (provisto)

Se libera **al inicio del evento**:

- Varios cientos de conversaciones telefónicas en **español**
- Caller humano real vs voz sintética
- **Estéreo, 8 kHz** — ch0 = caller, ch1 = agente
- Diverso a propósito: speakers, dispositivos, acentos, condiciones variadas

## Datasets de entrenamiento (externos)

### Núcleo

| Dataset | Qué es | Tamaño | Enlace |
| --------- | -------- | -------- | -------- |
| **ASVspoof 2019 LA** | Estándar de entrada. 19 sistemas TTS/VC | ~121k utterances | [DataShare](https://datashare.ed.ac.uk/handle/10283/3336) · [Kaggle](https://www.kaggle.com/datasets/hahunavth/asvspoof2019-la) |
| **ASVspoof 2021 LA** | Eval-only (reusa train de 2019) | 181,566 trials | [Zenodo 4837263](https://zenodo.org/records/4837263) |
| **ASVspoof 2021 DF** | Track deepfake con codec telefónico | — | [Zenodo 4835108](https://zenodo.org/records/4835108) |
| **ASVspoof 5** | Crowdsourced, ~2000 speakers, 32 algoritmos | Grande | [Zenodo 14498691](https://zenodo.org/records/14498691) · [HF](https://huggingface.co/datasets/jungjee/asvspoof5) |

### Diversidad de generadores

| Dataset | Qué aporta | Enlace |
| --------- | ------------ | -------- |
| **WaveFake** | 196h, 6 vocoders | [arXiv 2111.02813](https://arxiv.org/abs/2111.02813) |
| **Fake-or-Real (FoR)** | 195k+ utterances, 7 TTS | [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) |
| **CodecFake** | Deepfakes por codecs (EnCodec, SoundStream) | [GitHub](https://github.com/roger-tseng/CodecFake) |
| **MLAAD** | Multi-idioma (23), 70+ motores TTS | [arXiv 2401.09512](https://arxiv.org/abs/2401.09512) |

### In-the-wild

| Dataset | Qué aporta | Enlace |
|---------|------------|--------|
| **In-the-Wild** | Deepfakes reales de figuras públicas | [HF](https://huggingface.co/datasets/mueller91/In-The-Wild) |
| **Deepfake-Eval-2024** | Benchmark multimodal 2024 | [GitHub](https://github.com/nuriachandra/Deepfake-Eval-2024) |

### Post-RTC (más relevante para llamadas)

**RTCFake / RTC-SDD Challenge**

- ~400 horas de audio
- 10 tecnologías SOTA TTS/VC
- Audio por **7 plataformas reales** (Zoom, WeChat, Telegram, etc.)
- Subset clean + noisy
- Acceso: [formulario](https://docs.google.com/forms/d/e/1FAIpQLSdQlOPpi9WWx3RDlXwRhTVEWrGP0Y6FNAHolffgDdA5ESNdlQ/viewform) → [HF](https://huggingface.co/datasets/JunXueTech/RTCFake)

> ⚠️ RTCFake prohíbe datos externos y ensembles. Verificar si aplican al hackathon.
