# Where Hugging Face Fits: A Platform Comparison for R Programmers

> Research artifact for the companion book. Compares the major platforms an R
> programmer can use to **find, run, and deploy** ML/LLM models, and argues for a
> Hugging Face-first workflow that hands off to a managed platform (Microsoft
> Foundry) for governed production. Facts were gathered from current (June 2026)
> sources; key claims carry inline source URLs. A "could not verify" list closes
> the document.

A theme runs through every section below: **none of these platforms ships an
official R SDK.** All of them are reachable from R because they expose
OpenAI-compatible or plain REST endpoints that `httr2` can call — which is
exactly the niche `huggingfaceR` fills for the Hugging Face ecosystem. The book
should make this explicit: R is a first-class *client* of these services even
when it is not a first-class *SDK target*.

---

## 1. Hugging Face (Hub + Inference Providers) — the package's home base

**What it is.** The "GitHub of models": a hub of 2M+ open model weights,
datasets, and Spaces, plus two inference paths — serverless **Inference
Providers** (a router that fans out to ~18 compute partners behind one
OpenAI-compatible API) and **Inference Endpoints** (one-click dedicated
deployment of any open or private model on managed hardware).

**R access.** `huggingfaceR` via `httr2` — no Python. Chat/embeddings hit
`https://router.huggingface.co/v1/...`; classic tasks hit
`https://router.huggingface.co/hf-inference/models/{id}`. A single HF token
authenticates everything.

**Pricing.** Free serverless tier (rate-limited); Pro (~$9/mo) for higher
limits; Inference Endpoints billed per-instance/uptime. No markup framing — you
pay providers' rates through the router.

**Strengths.** Unmatched breadth of *open* models; discovery + datasets + Spaces
in one place; portability (download weights, self-host); the only ecosystem
where "find a model" and "fine-tune a model" are native.

**Weaknesses.** The serverless tier is a prototyping surface, not a governed
production gateway — no enterprise RBAC, private networking, data-residency
guarantees, or built-in observability/audit. Serverless availability is a
*curated subset* of the Hub, so not every model is callable without a dedicated
endpoint.

**Best-fit.** Discovery, prototyping, embeddings/classification/extraction from
R, and fine-tuning — the front of the pipeline.

---

## 2. Microsoft Foundry (formerly Azure AI Foundry, originally Azure AI Studio)

**Naming (important for the book).** Azure AI Studio (2023) → Azure AI Foundry
(renamed Ignite 2024) → **Microsoft Foundry** (renamed Ignite 2025, formalized in
the January 2026 Microsoft Product Terms). Docs distinguish "Microsoft Foundry"
(new) from "Microsoft Foundry (classic)"; portal is `ai.azure.com`. Recommend
"Microsoft Foundry (formerly Azure AI Foundry)" on first mention.
[SCHNEIDER IT](https://www.schneider.im/microsoft-foundry-the-new-name-for-azure-ai-foundry/),
[SAMexpert — Product Terms Jan 2026](https://samexpert.com/microsoft-product-terms-january-2026/),
[MS Learn — What is Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/what-is-foundry).

**What it is.** An enterprise platform to build, ground, deploy, govern, and
monitor AI apps/agents. A curated catalog ("over 1,900" models — Azure OpenAI,
Anthropic, Meta, Mistral, DeepSeek, xAI, Cohere, Hugging Face, NVIDIA), plus
Foundry Agent Service.
[MS Learn — Foundry Models overview](https://learn.microsoft.com/en-us/azure/foundry/concepts/foundry-models-overview).

**R access.** **No R SDK** (SDKs are Python/C#/JS/Java). R uses the REST API via
`httr2`/`jsonlite`. The OpenAI-compatible data plane
(`https://<resource>.services.ai.azure.com/openai/v1/chat/completions`) is the
simplest path; Hugging Face / open models deploy as **managed online endpoints**,
each exposing its own secure REST API. Auth via API key (`api-key` header) or
Microsoft Entra ID bearer token (enterprise-recommended; obtainable from R via
`AzureAuth`/`AzureRMR` or `az account get-access-token`).
[MS Learn — SDK overview](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/sdk-overview),
[MS Learn — Endpoints for Foundry Models](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/endpoints).

**Pricing.** Pay-as-you-go (serverless per-token; Global/Data Zone/Regional
SKUs), Provisioned Throughput Units (hourly $/PTU), PTU Reservations, and managed
compute (dedicated GPUs).
[MS Learn — Provisioned throughput billing](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/provisioned-throughput-billing).

**Strengths.** Observability (Foundry Control Plane → Azure Monitor; evaluations,
tracing, drift — GA 2026); governance (Compliance pane, Azure Policy/Defender/
Purview); content safety (prompt-injection/XPIA filters, AI Red Teaming Agent);
Entra RBAC; private networking + data residency (VNet, Private Link, CMK);
ISO/IEC 42001 certification.
[Foundry Control Plane](https://learn.microsoft.com/en-us/azure/foundry/control-plane/overview),
[Azure Blog — ISO/IEC 42001](https://azure.microsoft.com/en-us/blog/microsoft-azure-ai-foundry-models-and-microsoft-security-copilot-achieve-iso-iec-420012023-certification/).

**Weaknesses.** No R SDK; high complexity (classic vs new, many endpoint/auth
permutations); naming churn; Azure lock-in; cost vigilance needed (PTUs/GPUs
bill regardless of utilization); GPU quota friction.

**Best-fit.** The governed gateway between raw models and employee/customer apps
— strongest for regulated industries and Azure/M365 shops.

**HF integration (verified).** A dedicated **HuggingFace collection** in the
Foundry catalog exposes **11,000+ deployable HF models** via Azure ML managed
online endpoints; gated models use a `HuggingFaceTokenConnection` (`HF_TOKEN`);
models pass a five-stage curation/security pipeline. Origin: the Build 2025
HF–Microsoft partnership expansion.
[HF — Hugging Face on Microsoft Foundry](https://huggingface.co/docs/microsoft-azure/foundry/introduction),
[MS Learn — Deploy open-source models with managed compute](https://learn.microsoft.com/en-us/azure/foundry/how-to/deploy-models-managed).

---

## 3. GitHub Models — ⚠️ sunsetting

**Critical, time-sensitive finding.** On **June 16, 2026** GitHub announced
GitHub Models is **no longer available to new customers**; existing customers
continue "as usual" while it "moves toward full retirement," and GitHub points
new projects to **Azure AI Foundry**. No hard shutdown date was given.
[GitHub changelog, 2026-06-16](https://github.blog/changelog/2026-06-16-github-models-is-no-longer-available-to-new-customers/).
For a 2026 book: frame it as an *onboarding/historical* example, not a foundation
to build on.

**What it is.** A GitHub-branded, low-friction front door to models served on
**Azure AI infrastructure** — a free in-browser playground + an OpenAI-compatible
inference REST API (GA May 2025; paid tier June 2025).

**R access.** No R SDK. `httr2` POST to
`https://models.github.ai/inference/chat/completions`, `Authorization: Bearer
<PAT>` with a fine-grained PAT carrying **`models: read`** (or `$GITHUB_TOKEN` in
Codespaces/Actions). Catalog/embeddings endpoints also exist.
[GitHub REST — models inference](https://docs.github.com/en/rest/models/inference).

**Pricing.** Free, tightly rate-limited prototyping tier (e.g. GPT-4o ~10 req/min,
50 req/day); production requires opt-in paid usage metered via GitHub billing on
Azure-style per-token-unit economics.
[GitHub Models prototyping docs](https://docs.github.com/github-models/prototyping-with-ai-models),
[paid tier changelog](https://github.blog/changelog/2025-06-24-github-models-now-supports-moving-beyond-free-limits/).

**Strengths.** Near-zero friction for GitHub users; free prototyping;
OpenAI-compatible; trivial token story in CI.
**Weaknesses.** Being retired; tight free limits; smaller curated catalog than
HF; no R SDK.
**Best-fit.** Prototyping for developers already in GitHub — now a stepping stone
to Foundry (production) or HF (open weights).

---

## 4. Kaggle (the "Haggle" in the request — no such platform exists; Kaggle is meant)

A check found no notable ML platform named "Haggle" (only an unrelated
procurement agent). The intended platform is **Kaggle** (Google).

**What it is.** Google's data-science community: Competitions, a huge Datasets
repo, cloud **Notebooks** with free GPU/TPU, and **Kaggle Models** (a hub
aggregating partner, Keras, HF-integrated, and community models, with a Vertex AI
tie-in).
[Kaggle Models](https://www.kaggle.com/models),
[Using Vertex AI on Kaggle](https://www.kaggle.com/code/ryanholbrook/using-google-s-vertex-ai-on-kaggle/data).

**R access — unusually strong.** Kaggle Notebooks support **R as a native kernel
language** (Python or R backend; R Markdown via Script mode), and the Kaggle
Public API treats `r` as a first-class kernel language (`--language r`).
Community R packages exist (`RKaggle` on CRAN; `kaggler`, `KaggleR` on GitHub);
`kagglehub` is Python-only.
[How to use R in Kaggle](https://www.kaggle.com/getting-started/27082),
[kaggle-cli kernels docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
[CRAN RKaggle](https://cran.r-project.org/package=RKaggle).

**Pricing.** Free, Google-backed; approximate free quotas ~30 GPU h/week, ~20 TPU
h/week, 12 h max session (figures fluctuate).
[Kaggle TPU docs](https://www.kaggle.com/docs/tpu).

**Strengths.** Free GPU/TPU; huge datasets; strong learning/competition
ecosystem; genuine native R notebooks; HF + Vertex integration.
**Weaknesses.** **Not a production serving platform** (no deploy-an-endpoint
path — that's Vertex AI); quota/session caps; notebooks aren't always-on; R
tooling lags Python.
**Best-fit.** Learning, EDA, experimentation, competitions, free accelerators —
including hosted R with GPUs.
**Relation to HF.** Complementary, with an official bidirectional integration
("Use this model → Kaggle"; "Open in Hugging Face").
[HF blog — Kaggle integration](https://huggingface.co/blog/kaggle-integration).

---

## 5. ModelScope (Alibaba)

**What it is.** Alibaba's open-source "Model-as-a-Service" hub — widely described
as China's Hugging Face: models (notably **Qwen**), datasets, Studios/Spaces, and
**API-Inference**. 70,000+ models reported by mid-2025.
[ModelScope GitHub](https://github.com/modelscope/modelscope),
[Alibaba Cloud — English edition launch](https://www.alibabacloud.com/blog/alibaba-cloud-launches-english-language-version-of-open-source-ai-model-hub-modelscope_601320).

**R access.** No R SDK (Python SDK/CLI is primary), but there is an
**OpenAI-compatible REST inference API** at
`https://api-inference.modelscope.cn/v1/` (`/chat/completions`, Bearer token), so
R can call it directly with `httr2`. English edition at modelscope.ai.
[ModelScope API-Inference intro](https://www.modelscope.cn/docs/model-service/API-Inference/intro).

**Pricing.** Free hub; API-Inference free quota ~2,000 calls/day; Alibaba
Cloud-backed.

**Strengths.** Best source for Qwen and Chinese/multilingual models; large hub;
free OpenAI-compatible inference; reliable when HF is blocked.
**Weaknesses.** China-centric; Chinese-first docs; potential latency/access from
outside China; no R SDK; less Western-enterprise governance; token vs DashScope
key ambiguity.
**Best-fit.** Accessing Qwen/Chinese-origin models, and an HF fallback — from R
via REST.
**Relation to HF.** A regional analog/mirror; many models (Qwen) dual-published;
the recommended HF fallback.

---

## 6. OpenRouter

**What it is.** A unified API gateway/aggregator routing to 400+ models across
60–70+ providers behind one OpenAI-compatible endpoint and one bill — explicitly
a routing service, not a model trainer or hub. Default load-balancing +
provider/model failover.
[OpenRouter docs — models](https://openrouter.ai/docs/guides/overview/models),
[provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

**R access.** No official R SDK; OpenAI-compatible at
`https://openrouter.ai/api/v1/chat/completions`, `Authorization: Bearer
<OPENROUTER_API_KEY>` (optional `HTTP-Referer`/`X-Title` headers) — call from
`httr2` or any OpenAI-compatible R client (`ellmer`/`tidyllm`) by overriding the
base URL.
[OpenRouter quickstart](https://openrouter.ai/docs/quickstart).

**Pricing.** Prepaid credits, pay-per-token at passthrough provider rates (no
inference markup); a **5.5% platform fee on credit purchases** ($0.80 min; 5%
crypto); BYOK first 1M requests/mo free then 5%; ~25–29 free models with tight
limits.
[Platform fee announcement](https://openrouter.ai/announcements/simplifying-our-platform-fee),
[FAQ](https://openrouter.ai/docs/faq).

**Strengths.** One key/bill for many commercial + open models; OpenAI-compatible
drop-in; automatic failover; cost/latency routing; transparent passthrough
pricing; privacy controls (ZDR).
**Weaknesses.** Extra hop + credit fee; not a hub or fine-tuning platform
(BYO-model routing only); depends on upstream providers; lighter governance than
a full enterprise platform; free models unsuitable for production.
**Best-fit.** App developers wanting flexible multi-model access + resilience
without juggling provider accounts.
**Relation to HF.** Both are zero-markup routers, but on different ecosystems: HF
= open-weights hub + router + dedicated hosting; OpenRouter = thin routing/billing
layer over mostly commercial chat models, no hub/hosting/fine-tuning.

---

## 7. Comparison at a glance

| Platform | Primary role | R access | Open weights / self-host | Fine-tune | Enterprise governance & monitoring | Pricing shape |
|---|---|---|---|---|---|---|
| **Hugging Face** | Open model hub + inference | `huggingfaceR`/httr2, one token | ✅ download + self-host | ✅ native | ⚠️ via dedicated Endpoints; not a governed gateway | Free tier; Pro ~$9/mo; Endpoints per-uptime |
| **Microsoft Foundry** | Governed deploy/monitor/agents | httr2 (no R SDK); OpenAI-compatible REST | ✅ HF models via managed endpoints | ✅ (Azure ML) | ✅✅ RBAC, private net, residency, observability, ISO 42001 | PAYG / PTU / reservations / managed compute |
| **GitHub Models** ⚠️ retiring | Prototyping front-door (Azure-backed) | httr2 (no R SDK); PAT `models:read` | ❌ hosted only | ❌ | ⚠️ inherits Azure; limited | Free (tight) → opt-in metered |
| **Kaggle** | Learn / EDA / compete / free compute | ✅ native R notebooks + API | partial (download) | via notebooks (not serving) | ❌ not a serving platform | Free (quota'd) |
| **ModelScope** | Open hub (China/Qwen) | httr2 (no R SDK); OpenAI-compatible REST | ✅ download | ✅ | ❌ Western-enterprise gaps | Free hub; ~2k calls/day |
| **OpenRouter** | Multi-provider router | httr2 (no R SDK); OpenAI-compatible REST | ❌ routing only | ❌ | ⚠️ lighter (guardrails/ZDR) | Credits; passthrough + 5.5% top-up fee |

---

## 8. Synthesis: why Hugging Face → Microsoft Foundry is a sound architecture

The pipeline the book builds toward:

```
Hugging Face → find / fine-tune model → Microsoft Foundry
            → secure deployment + monitoring → employee/customer application
```

The argument is **division of labor between an open ecosystem and a managed
platform**, each doing what it is best at:

- **Hugging Face owns discovery, experimentation, and fine-tuning.** Its breadth
  of open weights, datasets, and Spaces — reachable from R via `huggingfaceR` —
  makes it the right place to *find the right model* and *adapt it to your data*.
  Crucially, the resulting model lands back on the Hub (or a registry) as a
  portable artifact you own. This is where an R team iterates fast and cheaply on
  the serverless API.

- **Microsoft Foundry owns governed production.** The very things the open
  serverless API deliberately doesn't provide — Entra RBAC, private networking,
  data residency, content-safety guardrails, audit logging, evaluations/tracing,
  drift monitoring, ISO 42001 compliance — are Foundry's core. An HF model
  deploys into Foundry as a managed online endpoint (the 11,000+ HF collection),
  gaining a secure, monitored, policy-bound front.

- **R stays the same at both ends.** Because Foundry managed endpoints are
  OpenAI-compatible, the same R code that prototyped against HF's serverless
  router can point at the governed Foundry endpoint by changing the
  `endpoint_url` (and the token/auth header). The application layer (a Shiny app,
  a Plumber API, a downstream product) then consumes that one governed endpoint.

**Second-order concerns this addresses for an enterprise R team:**

- *Governance & auditability:* who can call which model, with what data, logged
  where — answered by Foundry, not by a raw serverless key.
- *Data residency / PII:* regional/Data-Zone deployments and private networking
  keep prompts and completions within compliance borders.
- *Cost control:* prototype cheaply on HF serverless; commit to PTUs/reservations
  only once a workload's shape is known in Foundry.
- *Observability & drift:* production needs metrics, traces, and evaluation
  harnesses — Foundry publishes these to Azure Monitor.
- *Reproducibility & portability:* the fine-tuned artifact lives on the Hub, so
  you are not locked to a single inference vendor; you *choose* to deploy it in a
  governed place.

**Honest tradeoff.** The HF side maximizes openness, breadth, and portability but
under-provides governance; the Foundry side maximizes security/compliance/support
but adds Azure coupling, cost vigilance, and complexity. They are complementary,
not competing — and the alternatives above slot in as: GitHub Models (a
sunsetting prototyping front-door that itself points to Foundry), Kaggle (free
compute for experimentation/fine-tuning, not serving), ModelScope (an HF analog
for Qwen/Chinese models or as a fallback), and OpenRouter (a convenient
multi-vendor router for app-layer flexibility when governance needs are lighter).

---

## 9. Could not fully verify (flag before publishing)

- Several `learn.microsoft.com`, `github.blog`, `docs.github.com`, `openrouter.ai`,
  `kaggle.com`, and CRAN pages returned HTTP 403 to automated fetching; many
  claims rest on search-result excerpts quoting those primary pages plus
  corroborating accessible sources. Confirm exact wording/numbers at the cited
  URLs.
- **Foundry:** exact current Entra OAuth scope string (`https://ai.azure.com/.default`
  vs legacy Cognitive Services scope); "1,900 catalog" vs "11,000 HF collection"
  count different things — don't conflate.
- **GitHub Models:** exact `X-GitHub-Api-Version`, current production rate
  limits/per-token-unit price, and any firm shutdown date.
- **OpenRouter:** exact free-tier limits and model/provider counts drift; 5.5%/
  BYOK figures reconfirm on the live pricing page.
- **Kaggle:** GPU/TPU hour quotas fluctuate; RKaggle version/author read from
  metadata, not full CRAN page.
- **ModelScope:** ModelScope token vs DashScope key ambiguity; 2,000 calls/day
  quota from docs + secondary reviews.
- **R SDK absence** for Foundry/GitHub Models/OpenRouter/ModelScope is strongly
  supported but inherently hard to prove; no official R SDK was found for any.
