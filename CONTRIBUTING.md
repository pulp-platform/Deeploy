# Contribution Guide

We encourage submitting your issues and work in merge requests against the devel branch. Please understand that we are trying to maintain a consistent minimal quality standard.
Any and all merge requests you submit can only be accepted under the Apache 2.0 License.

## Overview

* The only way new features are accepted are by merge requests against the devel branch. Understand that we expect you to rebase your work against the devel branch if you submit merge requests.
* We encourage early draft merge requests to keep development transparent and avoid diverging efforts. Please submit your draft merge requests clearly labelled with "DRAFT:".
* We encourage refactoring. As code evolves, semantic concepts may change, and this is best addressed with refactoring. Please submit merge requests that implement refactoring with a label "REFACTOR:"
* We strongly encourage discussion on merge requests. Please comment on open merge requests, and keep it productive. The goal is to include feature ideas that are compatible with the Deeploy framework. Feedback for collaborators should include clear actionable items to improve the contribution.
* If a merge requests addresses a specific feature requests / bug, please reference it in the pull request.
* Deeploy is a research project. We do not expect a production level workflow, but we ask to add at the very least a proof of concept for any feature implementation. Similarly, if your merge request fixes a bug, please add a regression test for the error condition that was addressed.


## Style guide

Deeploy mainly consists of code implemented in C, Makefile, and Python. To facilitate efficient collaboration among users and contributors, it is important to maintain a consistent coding style. To achieve this, it is strongly recommend to use autoformatting tools with the provided configuration files. Additionally, the Continuous Integration (CI) system checks the adherence to the style guide for each pushed commit. Currently configuration for C using `clang-format` and for Python using `yapf` and `isort` are provided.

To recursively format all Python files run
```bash
$>	autoflake -i -r --remove-all-unused-imports --ignore-init-module-imports --exclude "*/third_party/**" ./
$>	yapf -ipr -e "third_party/" -e "install/" -e "toolchain/" ./
$>	isort --sg "**/third_party/*"  --sg "install/*" --sg "toolchain/*" ./
```

And for C files
```bash
$> python scripts/run_clang_format.py -e "*/third_party/*" -e "*/install/*" -e "*/toolchain/*" -ir --clang-format-executable=${LLVM_INSTALL_DIR}/bin/clang-format ./
```

Note that third party applications should not be formatted. You can alternatively also run
```
make format
```
to format all C and Python files.
