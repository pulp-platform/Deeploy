# Contribution Guide

We encourage submitting your issues and work in pull requests against the `devel` branch. Please understand that we are trying to maintain a consistent minimal quality standard.

## Overview

* The only way new features are accepted are by pull requests against the devel branch. We expect you to rebase your work against the devel branch if you submit pull requests.
* We encourage early draft pull requests to keep development transparent and avoid diverging efforts. Please submit your draft pull requests clearly labelled with "DRAFT:".
* We encourage refactoring. As code evolves, semantic concepts may change, and this is best addressed with refactoring. Please submit pull requests that implement refactoring with a label "REFACTOR:"
* We strongly encourage discussion on pull requests. Please comment on open pull requests, and keep it productive. The goal is to include feature ideas that are compatible with the Deeploy framework. Feedback for collaborators should include clear actionable items to improve the contribution.
* If a pull requests addresses a specific feature requests / bug, please reference it in the description.
* Deeploy is a research project. We do not expect a production level workflow, but we ask to add at the very least a proof of concept for any feature implementation. Similarly, if your pull request fixes a bug, please add a regression test for the error condition that was addressed.


## Changelog
All pull requests must include a concise and meaningful entry in the changelog under the unreleased section and category (added, changed, fixed, removed).
Additionally, add the title and link to the pull request in the list of pull requests. The changelog is located in `CHANGELOG.md` and should be updated in the following format:

```
## Unreleased (Planned Release Target: vx.x.x)
[...]

### List of Pull Requests
- Fix Linting in CI and Reformat C Files [#86](https://github.com/pulp-platform/Deeploy/pull/86)
[...]

### Addedd
- Reformatted all C files
[...]

### Fixed
- Fixed C-code linting stage in CI
[...]

### Removed
- Remove the link to the precompiled LLVM 12 in the `deeployRunner` for Snitch and in the CI.
[...]
```
## Style guide

Deeploy mainly consists of code implemented in C, Makefile, and Python. To facilitate efficient collaboration among users and contributors, it is important to maintain a consistent coding style. We use [pre-commit](https://pre-commit.com) with autoformatting tools to maintain this consistency. Configuration is provided for C using `clang-format` and for Python using `yapf` and `isort`.

### Setting up pre-commit

Install pre-commit (not included in `pyproject.toml`):
```bash
pip install pre-commit
```

Install the git hooks configured to run at the `pre-push` stage:
```bash
pre-commit install --hook-type pre-push
```

The hooks will automatically format your code before each push, ensuring it passes linting checks and CI validation.

To uninstall the git hooks:
```bash
pre-commit uninstall
```

You can also manually run formatting without pushing:
```bash
pre-commit run --all-files

# Or by running the Makefile target:
make format
```

## Licensing

Any and all pull requests you submit can only be accepted under the Apache 2.0 License. Every file needs to have an SPDX license header. We use the [reuse-tool](https://github.com/fsfe/reuse-tool) to check for the license header. You can use the same tool to add the license by calling it with the `annotate` command.
