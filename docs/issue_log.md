# Issue Log

## HuggingFace datasets compatibility error when loading Amazon Reviews 2023

- Severity: Medium
- Status: Open

### Symptom

```text
Dataset scripts are no longer supported, but found Amazon-Reviews-2023.py
```

### Impact

Amazon All_Beauty inspection did not complete; no inspection report was generated.

### Root Cause Hypothesis

The installed datasets version removed support for dataset loading scripts required by McAuley-Lab/Amazon-Reviews-2023.

### Next Diagnostic Step

Check installed datasets version with python3 and decide whether to use a compatible isolated environment or another official loading method.

Do not upgrade or downgrade datasets without user confirmation.
