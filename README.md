<div align="center">

# Jev-AR

**Arabic intent routing for RAG apps and assistants, in a single forward pass**

MSA · Emirati · Saudi · Arabic–English code-switching

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-atmaneayoub%2Fjev--ar-FFD21E)](https://huggingface.co/atmaneayoub/jev-ar)
[![Benchmark](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark-jev--ar--bench-F97316)](https://huggingface.co/datasets/atmaneayoub/jev-ar-bench)
[![Weights: CC BY-NC 4.0](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-6B7280)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-16A34A)](LICENSE)

</div>

---

Jev-AR reads a user message and **your list of routes** (a RAG app's knowledge bases, a helpdesk's queues, a
bot's intents) and picks the right one in a single forward pass. It returns calibrated probabilities and an
**escalate** flag for when it is unsure. It also ships with 60 built-in UAE + KSA government-service routes.

- **Your own routes, no retraining.** Pass any route list at request time: up to 60 short names, or about 15–20
  routes with one-line descriptions (fewer when the descriptions are long or in Arabic).
- **Gulf Arabic.** MSA, Emirati and Saudi dialects, and Arabic mixed with English.
- **Knows when it doesn't know.** A calibrated confidence and an escalation threshold fitted for 95% accuracy.
- **Small and fast.** 307M parameters: about 10 ms per message on a GPU. On a laptop CPU (8 threads), about 0.15 s
  with a short route list and about 0.45 s with all 60 government routes.

## Results

**health-rag**: one health-insurance RAG app with 6 routes (FAQ, escalate, benefits, exclusions, network check,
nearest in-network provider) and 20 questions across the four language slices. Each question is routed over the
English and the Arabic route list, so there are 40 decisions. Insurance and healthcare were not in the training
data.

| System | Score (of 40) | Order-robust* |
|:--|:--:|:--:|
| Jev 1.13 (TypeSafe, closed API) | 40 | ✓ |
| Qwen3.8-27B (27B LLM) | 40 | ✓ |
| **Jev-AR (307M)** | **38** | **✓** |

\* No answer changes when the route list is reversed. The two misses are one question in both languages, both
answered at low confidence, so they are flagged for escalation.

**UAE + KSA government services**: 4,974 frozen test questions written by a different model family, accuracy (%).

| System | 60 routes | 20 routes |
|:--|:--:|:--:|
| Qwen3.8-27B (LLM, list in prompt) | 90.0 | 94.0 |
| Jev 1.13 (TypeSafe) | 86.2 | 91.4 |
| **Jev-AR** | **83.4** | **90.0** |
| mmBERT fine-tuned as a classifier | 76.5 | 79.8 |
| mE5-base + logistic regression | 72.4 | 77.4 |

Confidence intervals, per-slice results and limitations are on the [model card](https://huggingface.co/atmaneayoub/jev-ar).

## Quick start

The weights are gated. [Request access](https://huggingface.co/atmaneayoub/jev-ar) (approval is automatic), then:

```bash
pip install torch transformers huggingface_hub
hf auth login
python jev_ar_route.py "ابي اجدد الاقامة"
```

`jev_ar_route.py` is a single, self-contained file (also included in the model repo).

### Route over your own list

```python
from jev_ar_route import JevAR

router = JevAR()  # downloads atmaneayoub/jev-ar on first use

routes = {
    "FAQ": "General how-to questions: claims process, insurance card, app, working hours, documents.",
    "Escalate to a human": "The user asks for a human agent, complains, or has an urgent problem.",
    "Benefits": "What the plan covers and how much: limits, co-payment, maternity, dental, optical.",
    "Nearest in-network provider": "Where the nearest in-network hospital or clinic is to a given place.",
}
router.route("ابغى اكلم احد من خدمة العملاء ضروري", routes)
# {'route': 'Escalate to a human', 'confidence': 0.93, 'escalate': False, 'probabilities': {...}}
```

### Government services, built in

```python
router.route("ابي اجدد الاقامة")
# {'route': 'تجديد الإقامة', 'id': 'renew_residence', 'category': 'residency', 'confidence': 0.78,
#  'escalate': False, 'in_scope': 0.998, 'probabilities': {...}}
```

**Tips:** Keep route names and descriptions short, since the whole list must fit in 512 tokens; the helper raises
an error rather than silently cutting options. Treat `escalate: True` as "send to a person or a fallback".
On a CPU, call `torch.set_num_threads(8)` (or your number of performance cores) first: PyTorch's default of one
thread per logical core can be several times slower on laptop CPUs.

## Reproduce the benchmark

```bash
python benchmark/run_health_rag.py
# main score: 38/40  (English list 19/20, Arabic list 19/20; flips when reversed: 0 / 0)
```

The government test set is on the [benchmark dataset](https://huggingface.co/datasets/atmaneayoub/jev-ar-bench).

## Limitations

- The training and test data are synthetic, written by LLMs, and not validated by native speakers.
- health-rag is small (20 questions), so a 38 vs 40 difference is not statistically significant.
- Yes/no "is X covered?" questions lean to an *exclusions*-type route. Test such questions on your own list.
- Calibration was fitted on the government list, so confidences on other lists are approximate.

## License

| Component | License |
|:--|:--|
| Model weights ([Hugging Face](https://huggingface.co/atmaneayoub/jev-ar)) | **CC BY-NC 4.0**: non-commercial use, testing and evaluation |
| Code in this repository | Apache-2.0 |
| Benchmark data | CC BY 4.0 |

**Commercial use** of the model needs a licence: contact **ayoutatmane23@gmail.com**. Third-party credits are in
[NOTICE](NOTICE).

## Citation

```bibtex
@misc{jevar2026,
  title        = {Jev-AR: Arabic Intent Routing for MSA, Gulf Dialects and Code-Switching},
  author       = {atmaneayoub},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/atmaneayoub/jev-ar}}
}
```
