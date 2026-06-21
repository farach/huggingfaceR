# Use Cases & Audience — Who Actually Wants huggingfaceR

> Status: planning document. Lives in `dev/book/` so it can be edited or deleted
> freely; it is not part of the built package, the pkgdown site, or CRAN.
>
> This document develops the *positioning* and *audience* thesis that Section 2
> of `book-outline.md` states briefly. It exists to answer a sharper question
> than "what can people do with Hugging Face?" — namely, **who would genuinely
> appreciate reaching Hugging Face from R, and why are they not the same people
> who would naturally reach for Python + Transformers?** Use it to inform the
> book's preface, Chapter 1 ("Why huggingfaceR"), and marketing/README framing.

---

## 1. The core thesis

A package like `huggingfaceR` does not just give R users a new tool — it
potentially **changes the audience** for Hugging Face.

The weak positioning is: *"R users can now call Hugging Face."* True, but not
compelling.

The strong positioning is:

> **Researchers, analysts, and data scientists who live in R can now bring
> modern AI models into their existing analytical workflows** — without leaving
> R, learning Python, or standing up ML infrastructure.

Sharpened into a one-line distinction (the thesis worth building the book
around):

> **Python users use Hugging Face to *build AI systems*.
> R users could use Hugging Face to turn AI into a *research instrument*.**

That distinction is where an R package has a real identity. The people who
appreciate an R interface are not necessarily the people who would naturally
reach for Python + Transformers — they are people for whom the model is a means
to an analytical end, not the product itself.

> Note: the use cases and audience profiles below are an **informed hypothesis**
> about where the package fits, not validated market research. Treat them as a
> positioning argument to test, refine with real users, and cite carefully if
> they make it into published prose.

---

## 2. The audience to target

If positioning `huggingfaceR`, **do not lead with ML engineers** — they already
live in Python. Lead with people who are R-native and analysis-first:

- **Researchers** — social science, economics, public policy, health,
  qualitative research.
- **Data analysts** — marketing analytics, customer insights, HR/people
  analytics.
- **Quantitative UX researchers** and **survey scientists**.
- **Data journalists.**
- **R-heavy organizations** — government, universities, healthcare research.

What unites them: they are comfortable in R and the tidyverse, are *less*
comfortable with ML infrastructure, and care about text and unstructured data
as evidence. They want insight, not a deployed model.

---

## 3. Use cases (ordered roughly by strength of fit)

Each use case below maps onto package capabilities the book already teaches
(`hf_embed`, `hf_classify`, `hf_classify_zero_shot`, `hf_summarize`,
`hf_ner`, clustering/topic/UMAP helpers, semantic search/nearest-neighbors).

### 3.1 Survey research at scale — *the strongest fit*

Survey researchers sit on open-ended responses, Likert scales, demographics, and
longitudinal panels, and often analyze the free text manually or with older NLP.

A natural R workflow:

```r
responses |>
  hf_embed(model = "sentence-transformers/...") |>
  cluster()
```

Use cases:

- discover themes in open-text responses
- find similar respondents
- identify emerging concerns
- compare sentiment across segments
- map qualitative comments to quantitative variables

Example — *"What are employees saying about AI adoption?"* Instead of manually
coding 50,000 comments:

```
comment → embedding model → topic clustering → segment analysis → ggplot
```

This is extremely natural for R users and is probably the clearest audience.

### 3.2 Social science / economics research — *the hidden niche*

Economists and social scientists are heavy R users, often less comfortable with
ML infrastructure, and increasingly interested in **text as data**.

- **Labor economics** — analyze job postings for skill requirements, AI
  exposure, occupational change, wage signals:
  `O*NET occupation + job description → embeddings → similarity → labor trends`.
- **Policy research** — analyze public comments, regulatory filings, speeches,
  legislative text.

### 3.3 Academic literature mining — *another killer use case*

Researchers have thousands of papers and weak keyword search. Instead:

```
14,500 abstracts → Hugging Face embeddings → UMAP → cluster → ggplot map of
research evolution
```

embed abstracts → find related papers → cluster fields → detect research trends.
Very aligned with R's data-visualization strengths.

### 3.4 Classification without building a model

Most analysts don't need to *train* a model — they need to *put things into
categories*. Zero-shot classification does this with no training data:

```r
hf_classify(
  text,
  labels = c("pricing", "support", "product feedback")
)
```

Examples: "I cannot find my invoice" → billing; "The app keeps crashing" →
reliability; "I love the new feature" → positive feedback. This can replace weeks
of manual coding.

### 3.5 Semantic search for analysts — *underappreciated*

Analysts have folders of PDFs, reports, transcripts, and notes where keyword
search is weak. Embeddings let *"find documents about workforce transformation"*
retrieve "organizational redesign", "AI-enabled teams", "automation anxiety" —
no exact-word match required.

```
documents → hf embeddings → vector store → search → analysis
```

### 3.6 Teaching AI concepts to statisticians — *pedagogical opportunity*

Many R users understand regression, clustering, visualization, and inference but
not modern AI. An R package makes the concepts tangible:

```r
embedding <- hf_embed(text)
plot(embedding)
```

Now they can *see* semantic space, similarity, clustering, and model behavior in
a vocabulary they already own.

### 3.7 Data journalism / storytelling — *R-native audience*

- take thousands of speeches → embed → identify topic/ideological shifts →
  visualize with ggplot
- analyze corporate earnings calls → map themes over time

The output is not a model; it is a **story**.

---

## 4. The bigger shift: the "AI analyst" workflow

The long-term interesting space is a shift in what the model is *for*:

```
Traditional:  Data → Model → Prediction
Emerging:     Data → AI transformation → Analysis → Insight
```

In the emerging pattern, AI is a transformation step that produces inputs for
ordinary analysis: summarize every row, classify every record, extract entities,
score text, generate features. R has always been strong at the final 80% —
statistics, visualization, reporting. Hugging Face provides new ways to create
the inputs.

---

## 5. How this informs the book

- **Preface / Chapter 1** should open with the thesis in §1 (research instrument
  vs. AI system) rather than "R can call HF."
- The **Northwind** running case study (a support-analytics program) already
  exercises §3.1, §3.4, and §3.5; the book can name the broader audiences in §2
  as "this is also you" callouts.
- The **applied interludes** (Anthropic Economic Index, OpenAI GDPval) are
  natural homes for the §3.2 (economics / text-as-data) framing.
- Consider a short, recurring **"Who this is for"** callout per part, mapping the
  capability to one of the §2 audiences — reinforcing identity over syntax.
- Keep the §1 note in mind: these are positioning hypotheses; validate with real
  users before stating them as fact in published prose.
