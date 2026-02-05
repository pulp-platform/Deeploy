# Deeploy Release Guide

This guide explains how to prepare and publish a Deeploy release to PyPI/TestPyPI using `uv`.

## Prepare the release
1. **Update the changelog** – Add a section for the new version under [CHANGELOG.md](../CHANGELOG.md) with the release date and noteworthy entries.
2. **Bump the version** – Use `uv` so the version in [pyproject.toml](../pyproject.toml) stays authoritative:
   ```bash
   uv version --bump major/minor/patch
   ```
3. **Verify the build locally** – This is optional but highly recommended:
   ```bash
   uv build
   uv run --isolated --no-project --with dist/*.whl python -c "import Deeploy"
   uv run --isolated --no-project --with dist/*.tar.gz python -c "import Deeploy"
   ```
4. **Commit and merge the changes** – Include the updated version and changelog in the commit. Once your commit reaches the `main` branch, it will be tagged.
5. **Deployment** – The publish workflow triggers on tags that start with `v` and match the version. Your package is now published congrats.
