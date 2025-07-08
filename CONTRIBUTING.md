# Contribution Guide

We encourage submitting your issues and work in pull requests against the `devel` branch. Please understand that we are trying to maintain a consistent minimal quality standard.
Any and all pull requests you submit can only be accepted under the Apache 2.0 License.

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
- Remove the link to the precompiled LLVM 12 in the `testRunner` for Snitch and in the CI.
[...]
```

## Style guide

Deeploy mainly consists of code implemented in C, Makefile, and Python. To facilitate efficient collaboration among users and contributors, it is important to maintain a consistent coding style. To achieve this, it is strongly recommend to use autoformatting tools with the provided configuration files. Additionally, the Continuous Integration (CI) system checks the adherence to the style guide for each pushed commit. Currently configuration for C using `clang-format` and for Python using `yapf` and `isort` are provided.

To recursively format all Python files run:
```bash
autoflake -i -r --remove-all-unused-imports --ignore-init-module-imports --exclude "*/third_party/**" .
yapf -ipr .
isort .
```

And for C files:
```bash
python scripts/run_clang_format.py -e "*/third_party/*" -e "*/install/*" -e "*/toolchain/*" -ir --clang-format-executable=${LLVM_INSTALL_DIR}/bin/clang-format ./
```

Note that third party applications should not be formatted. You can alternatively also run:
```bash
make format
```
to format all C and Python files.

### Pre-commit

Additionally, we provide the [pre-commit](https://pre-commit.com) configuration file which you can use to install github hooks that execute the formatting commands on your changes.

You will need to manually install pre-commit since it's not added as a dependency to the `pyproject.toml`:
```bash
pip install pre-commit
```

The configuration sets the default stage for all the hooks to `pre-push` so to install the git hooks run:
```bash
pre-commit install --hook-type pre-push
```
The hooks will run before each push, making sure the pushed code can pass linting checks and not fail the CI on linting.

If you change your mind and don't want the git hooks:
```bash
pre-commit uninstall
```

_Note:_ This configures only the python formatting git hooks. The c formatting is not supported at the moment.
