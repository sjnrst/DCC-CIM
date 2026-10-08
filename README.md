# DCC-CIM

Public code release for **Diagnostic Confidence Control for Candidate Indicator Mining with Large Language Models**.

`DCC-CIM` is short for **Diagnostic Confidence Control for Candidate Indicator Mining**.

## What is included

This repository contains the core implementation used for:

- diagnostic feature extraction for candidate indicators;
- Consensus-v2 construction from the full unlabeled candidate pool;
- document-level cross-validated seed-alignment confidence modeling;
- post-generation binary / three-way filtering evaluation;
- open-world stability routing and post-processing;
- consistency checks for the main intermediate and final outputs.

## Repository structure

```text
DCC-CIM/
├── src/
│   ├── 01_extract_diagnostic_features.py
│   ├── 02_build_consensus_v2.py
│   ├── 03_train_seed_alignment_model.py
│   ├── 04_evaluate_filtering_and_routing.py
│   └── 05_postprocess_open_world_routes.py
├── validation/
│   ├── validate_feature_tables.py
│   ├── validate_consensus_v2.py
│   └── validate_filtering_results.py
├── config/
│   └── model_sources.example.txt
├── .gitignore
├── requirements.txt
└── README.md
```

## Data and privacy

The public release intentionally **does not include**:

- source documents;
- annotations or adjudication records;
- candidate-level datasets;
- model responses or generated outputs;
- intermediate CSV/XLSX/JSONL files;
- local paths, API keys, credentials, logs, or personal information.

The scripts expect private experimental inputs to be provided locally. By default, raw model outputs used to reconstruct Consensus-v2 are expected under:

```text
private_inputs/model_outputs/
```

and generated intermediate artifacts are written under:

```text
artifacts/
```

Both locations are excluded by `.gitignore`.

## Environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Notes on reproducibility

This repository releases the methodological code while withholding private or non-redistributable research data. Exact reproduction of the reported numerical results therefore requires the corresponding experimental inputs, which are not part of this public package.

## Citation

Citation metadata can be added after the paper is formally published.
