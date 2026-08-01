## What this PR does

<!-- One or two sentences. What changed and why. -->

## Type of change

- [ ] New feature (dataset loader, training script, preprocessing module, etc.)
- [ ] Bug fix
- [ ] Config update (params.yaml, dvc.yaml, requirements.txt)
- [ ] Documentation (README, model_card, comments)
- [ ] Refactor / cleanup

## DVC changes

- [ ] No DVC changes in this PR
- [ ] Ran `dvc repro` and committed updated `dvc.lock`
- [ ] Ran `dvc push` after repro

## Checklist before requesting review

- [ ] Code runs end to end on a clean Colab session (not just my machine)
- [ ] No `data/` `results/` or `.pth` files staged (`git status` is clean of these)
- [ ] No hyperparameters hardcoded in `.py` files (everything in `params.yaml`)
- [ ] `python -m pytest tests/ -v` passes with no failures
- [ ] Inline comments added to all major code blocks
- [ ] Branch is up to date with `develop` (`git pull origin develop` before pushing)

## What the reviewer should check

<!-- Tell the reviewer specifically what to look at. -->

## Related files changed

| File | What changed |
|---|---|
|  |  |

## Notes for reviewer

<!-- Anything tricky, a known issue, or context the reviewer needs. -->
