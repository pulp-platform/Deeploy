# Changelog

## Unreleased

## Fix Generic Softmax Kernel

### Fixed
- Fix broken softmax kernel for generic platform ([#2](https://github.com/pulp-platform/Deeploy/pull/2)).

## Minor CI and Readme Improvements

### Added
- Improved README with more detailed `Getting Started` section, a section listing related publications, and a list of supported platforms.
- Schedule a CI run every 6 days at 2AM CET to refresh the cache (it expires after 7 days if unused).
### Fixed
- Update the link of the Docker container used to run the CI with the Docker published by this repo instead of my fork.
- Add a retry on timeout step for large network tests. This is a temporary fix to address the sporadic freeze happening at the compilation stage, see [this issue](https://github.com/pulp-platform/Deeploy/issues/9).
