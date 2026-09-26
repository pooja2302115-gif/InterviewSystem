This directory contains generated, cleaned, normalized, and deduplicated datasets. Do not edit these files by hand.

Run the Phase 2 cleaner from the repository root:

```bash
python backend/training/prepare_data.py
```

The command writes `train.jsonl`, `validation.jsonl`, and `test.jsonl` here, along with `cleaning_report.json`. Each accepted record keeps the original fields and adds:

- `source_split`: the source partition.
- `text`: a stable `Category / Topic / Difficulty / Question / Answer` template, with optional example, code, and complexity sections.

The report records malformed rows, schema errors, duplicate IDs, and duplicate rendered content. Any schema or malformed-row error makes the command exit with status 1.
