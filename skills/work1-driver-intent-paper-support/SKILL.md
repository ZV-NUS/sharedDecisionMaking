---
name: work1-driver-intent-paper-support
description: Read only the scoped Work1 driver decision and control-intent prediction implementation in the TASE02 repository to support IEEE paper writing. Use this skill when writing or verifying the method, formulas, dataset description, model architecture, loss design, or claim-evidence alignment for Work1.
---

# Work1 Driver Intent Paper Support Skill

## 1. Purpose

Use this skill to inspect and summarize only the code relevant to **Work1: driver decision and control-intent prediction**.

The goal is to support IEEE Transactions manuscript writing for the driver-intent prediction part, including:

- highD-based traffic state representation,
- driver decision prediction,
- driver control-intent prediction,
- Transformer-based historical sequence encoding,
- future maneuver event prediction,
- future speed and steering prediction,
- training objective design,
- evaluation metric interpretation,
- claim-code alignment for manuscript writing.

This skill is intentionally narrow. It prevents unnecessary reading of large datasets, checkpoints, logs, generated outputs, figures, web demos, and videos.

## 2. Repository Root

The repository root is:

```text
<PROJECT_ROOT>
```

All paths in this skill are relative to this root unless written as absolute paths.

## 3. Scope of Use

Use this skill only for paper-writing or method-verification tasks related to Work1, including:

- driver-intent prediction method description,
- Section III-A / III-B manuscript support,
- model architecture explanation,
- input-output definition,
- highD-derived label construction,
- inverse-kinematics-derived steering label explanation,
- loss function explanation,
- evaluation metric explanation,
- figure caption writing for the Work1 model,
- claim-evidence alignment between the paper text and Work1 code.

Do not use this skill to inspect Work2 machine policy, Work3 trust/authority, Work4 RL/MPC, closed-loop validation, videos, web visualization, or generated scenario outputs.

## 4. Allowed Files

Read only the following files by default.

### 4.1 Core Work1 Model Code

```text
<PROJECT_ROOT>\src\models\human_intent_transformer.py
```

Use this file to verify:

- Transformer encoder architecture,
- positional encoding,
- input feature fusion,
- risk/interaction context usage,
- prediction heads,
- forward-pass outputs,
- tensor shapes when available from code.

### 4.2 Core Work1 Dataset Code

```text
<PROJECT_ROOT>\src\data\intent_dataset.py
```

Use this file to verify:

- dataset class definitions,
- HDF5/NPZ sample loading interface,
- input tensor keys,
- target tensor keys,
- returned batch structure,
- dataset-level assumptions.

### 4.3 Work1 highD Preprocessing Code

```text
<PROJECT_ROOT>\src\data\highd_preprocess.py
```

Use this file only when manuscript writing needs to verify:

- frame-wise intent label construction,
- lane-change label definition,
- future decision sequence construction,
- lane-change count and return-flag construction,
- steering estimation by inverse kinematics,
- speed and acceleration derivation,
- neighbor feature construction,
- risk feature construction.

### 4.4 Work1 Training Script

```text
<PROJECT_ROOT>\scripts\train_human_intent_transformer.py
```

Use this file to verify:

- model initialization,
- training loop,
- loss terms,
- loss weights,
- class balancing,
- optimizer,
- validation/test evaluation calls,
- checkpoint saving logic.

### 4.5 Dataset Building Scripts

Read these files only if the manuscript paragraph concerns dataset construction or data packaging.

```text
<PROJECT_ROOT>\scripts\preprocess_highd.py
<PROJECT_ROOT>\scripts\build_intent_dataset.py
<PROJECT_ROOT>\scripts\pack_intent_hdf5.py
```

Use them to verify the workflow only. Do not follow them into actual datasets.

### 4.6 Work1 Configuration Files

```text
<PROJECT_ROOT>\configs\human_intent_transformer.yaml
<PROJECT_ROOT>\configs\human_intent_transformer_smoke.yaml
<PROJECT_ROOT>\configs\build_intent_dataset.yaml
<PROJECT_ROOT>\configs\preprocess_highd.yaml
```

Use these files to verify:

- history length,
- prediction horizon,
- model hyperparameters,
- loss weights,
- training settings,
- dataset split paths,
- preprocessing parameters.

## 5. Optional Small Result Files

Do not read result files by default.

Read the following files only if the user explicitly asks for Work1 quantitative results or result-table support:

```text
<PROJECT_ROOT>\checkpoints\human_intent_transformer\test_metrics.json
<PROJECT_ROOT>\checkpoints\human_intent_transformer\history.json
<PROJECT_ROOT>\checkpoints\human_intent_transformer_smoke\test_metrics.json
<PROJECT_ROOT>\checkpoints\human_intent_transformer_smoke\history.json
```

These files are small JSON files and may be used to support result-table writing.

Never read model weights:

```text
<PROJECT_ROOT>\checkpoints\human_intent_transformer\best.pt
<PROJECT_ROOT>\checkpoints\human_intent_transformer_smoke\best.pt
```

## 6. Do Not Read

Do not read, summarize, scan, or open the following unless the user explicitly expands the scope:

- raw highD data,
- HDF5 datasets,
- processed dataset shards,
- large `.npz` sample files,
- model weights,
- checkpoint binaries,
- tensorboard files,
- logs,
- generated figures,
- generated outputs,
- generated web pages,
- generated JavaScript visualization data,
- videos,
- draw.io files,
- large result folders,
- Work2 machine policy files,
- Work3 trust/authority files,
- Work4 RL/MPC/controller files.

Specifically avoid:

```text
<PROJECT_ROOT>\data\
<PROJECT_ROOT>\outputs\
<PROJECT_ROOT>\logs\
<PROJECT_ROOT>\checkpoints\*.pt
<PROJECT_ROOT>\checkpoints\*.pth
<PROJECT_ROOT>\checkpoints\*.ckpt
<PROJECT_ROOT>\outputs\**\*.mp4
<PROJECT_ROOT>\outputs\**\*.png
<PROJECT_ROOT>\outputs\**\*.jpg
<PROJECT_ROOT>\outputs\**\*.svg
<PROJECT_ROOT>\outputs\**\*.html
<PROJECT_ROOT>\outputs\**\*.js
<PROJECT_ROOT>\outputs\**\*.drawio
```

Also avoid these module folders unless the user explicitly asks for cross-module integration:

```text
<PROJECT_ROOT>\src\policies\
<PROJECT_ROOT>\src\trust\
<PROJECT_ROOT>\src\rl\
<PROJECT_ROOT>\src\control\
<PROJECT_ROOT>\src\envs\
```

## 7. Reading Procedure

When this skill is used, follow this order:

1. Read `configs\human_intent_transformer.yaml`.
2. Read `src\data\intent_dataset.py`.
3. Read `src\models\human_intent_transformer.py`.
4. Read `scripts\train_human_intent_transformer.py`.
5. Read `src\data\highd_preprocess.py` only if label construction, steering estimation, or highD preprocessing is part of the writing task.
6. Read dataset-building scripts only if the user asks about preprocessing workflow.
7. Read Work1 result JSON files only if the user explicitly asks for quantitative result support.

Do not recursively inspect the repository. Use the allowed file list as a strict whitelist.

## 8. If a Path Is Missing

If one of the allowed files is missing:

1. Report the missing path.
2. Do not search the whole repository.
3. Use `rg --files` only with a narrow pattern, for example:

```powershell
rg --files <PROJECT_ROOT> | rg "human_intent|intent_dataset|highd_preprocess|train_human|preprocess_highd|build_intent|pack_intent|human_intent_transformer.yaml"
```

4. Ask the user before expanding to unrelated directories.

## 9. Output Style

When summarizing Work1 for paper writing, organize the response into the following blocks.

### 9.1 Code-Supported Facts

Only include information verified from allowed files.

Examples:

- model inputs,
- model outputs,
- model architecture,
- prediction heads,
- training losses,
- dataset keys,
- label definitions.

### 9.2 Paper-Writing Interpretation

Explain how the code should be described in IEEE Transactions style.

Examples:

- "The driver-intent predictor encodes historical ego and interaction states using a Transformer encoder."
- "The model performs multi-task prediction of maneuver class, event timing, speed intent, and steering intent."
- "The steering target should be described as an inverse-kinematics-derived label rather than a directly measured steering-wheel angle."

### 9.3 Manuscript-Ready Formulas

Provide notation for:

- input traffic state,
- historical sequence encoding,
- Transformer feature representation,
- decision prediction,
- future event prediction,
- future control-intent prediction,
- multi-task loss.

Mark formulas as interpretation if they are not explicitly written in code.

### 9.4 Claim-Evidence Alignment

Use this format:

```text
Claim:
Evidence:
Status: supported / partially supported / requires validation
```

### 9.5 Unsupported Claims

List claims that cannot be supported from the allowed files.

Examples:

- "The model is state-of-the-art."
- "The method guarantees safety."
- "The model was validated in closed-loop control."
- "The steering labels are measured steering-wheel angles."

## 10. Recommended Paper Terminology

Prefer:

- driver decision and control-intent prediction,
- risk-aware driver intent prediction,
- highD-derived traffic state representation,
- history-to-future sequence modeling,
- Transformer-based temporal encoding,
- multi-task prediction heads,
- future maneuver event prediction,
- event-time prediction,
- future speed intent,
- future steering intent,
- inverse-kinematics-derived steering label,
- frame-wise maneuver label,
- driver decision probability,
- future control-intent sequence.

Avoid unless explicitly supported:

- state-of-the-art,
- guaranteed safety,
- real-time deployment,
- end-to-end autonomous driving,
- cognitive driver model,
- causal driver reasoning,
- trust-aware driver model,
- closed-loop shared control,
- reinforcement learning,
- MPC,
- machine intent generation.

These avoided terms belong to later work packages or require evidence outside Work1.

## 11. Manuscript Use Cases

Use this skill to support:

- Work1 method subsection,
- Section III-A / III-B driver-intent prediction writing,
- dataset and label construction paragraph,
- model architecture paragraph,
- loss function paragraph,
- evaluation metric paragraph,
- figure caption for Work1 architecture,
- notation table entries,
- claim-evidence table,
- reviewer response about Work1 implementation.

## 12. Expected Deliverables

Depending on the user request, produce one of the following.

### A. Work1 Code Summary

Summarize:

- input representation,
- model architecture,
- output heads,
- target labels,
- loss functions,
- evaluation metrics,
- manuscript-ready description.

### B. Work1 Method Draft

Write a polished IEEE-style method subsection.

### C. Claim-Evidence Table

Map manuscript claims to code files.

### D. Formula Support

Provide compact equations for:

- input encoding,
- Transformer prediction,
- event prediction,
- speed/steering intent prediction,
- multi-task training loss.

### E. Limitations for Writing

State what cannot be claimed from Work1 code alone.

## 13. Minimal Response Template

When asked to inspect Work1, respond using:

```text
I inspected only the allowed Work1 files.

Code-supported facts:
...

Paper-writing interpretation:
...

Claim-evidence alignment:
...

Unsupported or not-yet-verified claims:
...
```

## 14. Important Constraints

- Do not read large data files.
- Do not read model weights.
- Do not read generated videos or web demos.
- Do not inspect Work2, Work3, or Work4 unless explicitly requested.
- Do not claim full shared-driving system performance from Work1 alone.
- Do not claim steering labels are measured steering-wheel angles; describe them as inverse-kinematics-derived or estimated steering labels if supported by code.
- Do not report numerical performance unless the user explicitly permits reading the small Work1 result JSON files.
- Do not use result numbers from memory if the corresponding result file has not been read in the current task.
- Do not merge Work1 with trust, authority, RL, or MPC modules unless the user asks for cross-module integration.

## 15. Quick Checklist Before Answering

Before giving the final response for a Work1 paper-support task, check:

- Did I read only allowed files?
- Did I avoid HDF5, raw data, outputs, videos, and checkpoints?
- Did I distinguish code-supported facts from writing interpretation?
- Did I avoid unsupported claims such as state-of-the-art or guaranteed safety?
- Did I identify whether quantitative results were actually read or not?
- Did I keep Work1 separate from Work2/Work3/Work4?


