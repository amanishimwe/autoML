# matezaML

Open-source machine learning toolkit for low-resource environments.

## What This Project Is About

`matezaML` is designed for real-world data conditions that are common across Africa and other developing regions: small datasets, missing values, noisy records, class imbalance, and limited computing infrastructure.

Most mainstream ML libraries assume ideal conditions (large datasets, GPU/cloud access, and highly curated data). This project focuses on practical machine learning that can run on standard laptops, work offline, and still produce reliable, explainable, and fair outcomes.

The goal is to make machine learning more accessible to practitioners who may not have deep ML engineering backgrounds while still supporting robust workflows for research and policy applications.

## Why matezaML Exists

`matezaML` addresses common gaps between mainstream assumptions and local realities:

- Small datasets instead of millions of rows
- Standard local machines instead of GPU/cloud infrastructure
- Incomplete or noisy data instead of clean benchmark datasets
- Severe class imbalance in critical use cases
- Need for explainability and fairness in public-impact decisions

## Core Focus Areas

The toolkit is organized around practical modules for end-to-end work:

- **Preprocessing**: missing values, feature scaling, categorical encoding, outlier handling
- **Model training**: lightweight models suitable for smaller datasets
- **AutoML**: automatic model selection and tuning for non-expert users
- **Evaluation**: metrics beyond accuracy for imbalanced scenarios
- **Fairness**: demographic parity checks for responsible AI use
- **Explainability**: model interpretation support for transparency

## Who It Supports

`matezaML` is built for:

- African researchers and students
- National statistical offices
- Government institutions and policy teams
- Universities and research labs
- NGOs and social-impact organizations
- International development partners

## Project Structure

Python package and R package are included in the same repository to support both communities.

Canonical Python architecture:

```text
matezaML/
  matezaML/
    __init__.py
    _version.py
    types.py
    pipeline.py
    preprocessing/
      __init__.py
      imputer.py
      scaling.py
      encoding.py
      outliers.py
    models/
      __init__.py
      baseline.py
    evaluation/
      __init__.py
      classification.py
      report.py
    fairness/
      __init__.py
      demographic_parity.py
      report.py
  tests/
  examples/
```

Module responsibilities:

- `preprocessing`: missing-value handling, scaling, encoding, and outlier capping
- `models`: lightweight baseline estimators and model factories
- `evaluation`: imbalance-aware classification metrics and reporting helpers
- `fairness`: demographic parity metrics and group-level fairness reports
- `pipeline`: simple, sklearn-like end-to-end pipeline orchestration

## Vision

`matezaML` aims to become a global open-source toolkit for machine learning in low-resource environments, enabling trustworthy and practical AI where resources are constrained but decisions are high-impact.

## Contributing

Contributions are welcome.

Steps:

1. Fork repository
2. Create feature branch
3. Write tests
4. Submit pull request

## License

MIT License

## Author

Alban @Cofounder at Mateza
a(dot)manishimwe(a)mateza(dot)rw
