# Data Structure

## Included Data

The release includes:

```text
outputs/shared_authority_validation/shared_authority_rollouts.js
```

This lightweight rollout file is used for demo visualization, paper plotting, and DIL traffic injection.

## Not Included

The full raw highD dataset is not included due to dataset licensing and size.

Large processed datasets, checkpoints, logs, and videos are also excluded from GitHub.

## Local highD Path

Use an environment variable:

```powershell
$env:HIGHD_ROOT="D:\datasets\highD"
```

or copy:

```text
configs/local.example.yaml
```

to:

```text
configs/local.yaml
```

and edit the local dataset paths.

