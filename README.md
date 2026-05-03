# Company Fit Check

`company-fit-check` is a Python + LangGraph workflow for turning a CV PDF and a
free-form user prompt into:

- structured company search criteria
- structured matching axes
- a discovered company list
- scored company matches
- a CSV export for the final result

The current implementation is optimized for an interactive clarification loop.
It is not a generic recruiter tool or a general search engine. It is a
workflow for:

1. reading a CV safely
2. interpreting the user's intent
3. separating deterministic company filters from softer matching dimensions
4. discovering companies with the deterministic filters
5. scoring those companies on the softer dimensions

## Core Rule

The workflow is built around one separation rule:

- `company_search_criteria` contain company-side facts with an absolute or
  near-absolute answer
- `axes` contain everything that requires interpretation, investigation,
  assumptions, or probabilistic judgment

Examples of company search criteria:

- location
- company size
- industry or domain
- company stage
- role family
- work mode

Examples of axes:

- English-working environment
- career-switch openness
- compensation fit
- seniority fit
- culture fit
- transition friendliness

Search criteria are used only for company discovery. They are not supposed to
reappear as scoring dimensions later in the flow.

## Architecture

The backend is organized into four layers:

1. `graph/`
   Orchestration, nodes, and resume routing.
2. `services/`
   CV extraction, PII masking, simplification, prompt interpretation, company
   discovery, scoring, MLflow logging, and CSV export.
3. `interfaces/chainlit/`
   Thin UI adapter for chat sessions.
4. `models/`
   Typed state and artifact models shared across the workflow.

Important files:

- [workflow.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/graph/workflow.py)
- [nodes.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/graph/nodes.py)
- [routing.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/graph/routing.py)
- [prompt_interpretation.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/services/prompt_interpretation.py)
- [company_discovery.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/services/company_discovery.py)
- [company_scoring.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/services/company_scoring.py)
- [interpretation_validation.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/services/interpretation_validation.py)
- [result_exports.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/services/result_exports.py)

## Workflow

The runtime graph is:

```text
entry
  -> extract_and_mask_cv
  -> validate_pii_masking
  -> simplify_cv
  -> interpret_prompt
  -> validate_interpretation
      -> search_companies
      -> or stop for clarification
  -> score_companies
  -> end
```

There is one alternate clarification path:

```text
entry
  -> refine_company_search
  -> search_companies
```

The actual node responsibilities are:

1. `extract_and_mask_cv`
   Extract text from the PDF, then mask PII locally.
2. `validate_pii_masking`
   Fail if masking did not complete or produced empty output.
3. `simplify_cv`
   Use Azure OpenAI to normalize the masked CV into simpler plain text.
4. `interpret_prompt`
   Use Azure OpenAI to split the user request into `company_search_criteria`
   and `axes`.
5. `validate_interpretation`
   Decide whether interpretation is usable or whether the user needs to clarify
   missing / weak axis or CV context.
6. `search_companies`
   Use only `company_search_criteria` to discover companies.
7. `refine_company_search`
   If discovery returned zero companies, use the user's clarification to revise
   only the company search criteria.
8. `score_companies`
   Score discovered companies using only the simplified CV and `axes`.

## Clarification Model

The workflow supports two clarification targets:

- `interpretation`
- `company_search`

Interpretation clarification is used when:

- the axes are unclear
- there are too many axes
- axis descriptions are missing
- the CV does not contain enough information to score the requested axes

Company-search clarification is used when:

- the current deterministic search filters return zero companies

Clarification resumes are routed by `clarification_target` in
[routing.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/graph/routing.py).

## State Model

The in-memory workflow state is defined in
[state.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/models/state.py).

The key fields are:

- `input`
- `masked_cv_text`
- `simplified_cv_text`
- `company_search_criteria`
- `axes`
- `companies`
- `company_scores`
- `pending_clarification_message`
- `latest_clarification_response`
- `clarification_target`
- `session_status`
- `error`

Session statuses:

- `running`
- `needs_clarification`
- `completed`
- `failed`

## Outputs

When the workflow completes successfully, it produces:

- discovered companies with metadata and discovery reason
- axis-level scores per company
- one overall score per company
- a CSV artifact

The CSV is generated by
[result_exports.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/services/result_exports.py).
Its columns are:

- `company_name`
- `website_or_linkedin`
- `industry`
- `company_size`
- `discovery_reason`
- `overall_score`
- one `axis_*_..._score` column per axis

## Privacy Model

The workflow is intentionally conservative about CV handling:

- PDF text extraction happens locally
- PII masking happens locally
- if masking fails, the workflow fails before downstream LLM usage

The app still assumes the user should mask sensitive content manually when
possible. Automatic masking is a safeguard, not a formal privacy guarantee.

## LLM Usage

Azure OpenAI is used for:

- CV simplification
- prompt interpretation
- interpretation clarification assessment
- zero-result search clarification
- company search refinement
- company discovery
- company scoring

The current code assumes a configured Azure chat deployment for all of these
steps.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project requires Python `>= 3.11`.

Optional but recommended for stronger Presidio entity detection:

```bash
python -m spacy download en_core_web_lg
```

## Configuration

Runtime configuration is loaded from `.env` and the current process
environment by [config.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/config.py).

### Azure OpenAI

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_TEMPERATURE=0
AZURE_OPENAI_MAX_TOKENS=7000
```

### MLflow

MLflow logging is optional. If configured, the app persists run metadata,
clarification artifacts, workflow artifacts, discovered companies, and final
scoring artifacts.

```bash
MLFLOW_TRACKING_URI=file:///absolute/path/to/company-fit-check/.mlruns
MLFLOW_EXPERIMENT_NAME=company-fit-check
MLFLOW_ARTIFACT_ROOT=wasbs://your-container@your-account.blob.core.windows.net/company-fit-check
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net
```

Notes:

- local tracking defaults to `.mlruns` if `MLFLOW_TRACKING_URI` is not set
- artifact persistence is considered configured only when both
  `MLFLOW_ARTIFACT_ROOT` and `AZURE_STORAGE_CONNECTION_STRING` are present
- the helper script [start_mlflow_ui.sh](/home/rushenkottie/Desktop/company-fit-check/start_mlflow_ui.sh)
  starts the MLflow UI against the configured backend store

## Run The UI

The project includes a minimal Chainlit interface in
[app.py](/home/rushenkottie/Desktop/company-fit-check/src/company_fit_check/interfaces/chainlit/app.py).

Start it from the repository root:

```bash
chainlit run src/company_fit_check/interfaces/chainlit/app.py
```

Interaction contract:

1. first message must include exactly one PDF attachment and a prompt
2. later clarification turns are plain text replies in the same chat
3. completed runs return a summary plus a CSV attachment

The UI state is temporary and in-memory for the current chat session.

## Programmatic Usage

### Start a new run

```python
from pathlib import Path

from company_fit_check.graph.workflow import create_initial_state, run_workflow
from company_fit_check.models.input import UserInput

state = create_initial_state(
    UserInput(
        cv_pdf_bytes=Path("cv.pdf").read_bytes(),
        prompt="I am looking for embedded electronics companies in Belgium that operate in English.",
    )
)

state = run_workflow(state)
```

### Resume after clarification

```python
from company_fit_check.graph.workflow import apply_clarification

state = apply_clarification(
    state,
    user_response="No prior embedded experience except university projects.",
)
```

Possible results after each invocation:

- `session_status == "completed"`
- `session_status == "needs_clarification"`
- `session_status == "failed"`

If clarification is needed, the state contains:

- `pending_clarification_message`
- `clarification_target`

## Current Limits

The current implementation intentionally keeps some limits simple:

- maximum 4 axes
- maximum 5 clarification attempts for interpretation
- maximum 5 clarification attempts for company search refinement
- in-memory chat session state
- no public API layer
- no deployment / container / CI setup in repo

## Non-Goals In Current Repo

The repository does not currently include:

- a production API service
- persistence for user sessions outside the current process
- a ranking explanation UI beyond the final CSV and completion summary
- feedback learning or post-run preference tuning
- a formal retrieval backend or company database

## Summary

This codebase is a clarification-first company matching workflow:

- deterministic company filters are extracted once and used only for discovery
- softer, investigational fit dimensions become axes
- companies are discovered first
- those companies are scored later only on the axes

That separation is the main architectural contract of the project.
