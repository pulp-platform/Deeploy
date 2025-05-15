# Store old diff
GIT_OLD_DIFF="$(git diff --name-only refs/remotes/origin/HEAD)"

# Run formatting
echo "Running clang-format..."
python scripts/run_clang_format.py -e "*/third_party/*" -e "*/install/*" -e "*/toolchain/*" -ir ./ scripts --clang-format-executable=${LLVM_INSTALL_DIR}/bin/clang-format

echo "Running autoflake..."
autoflake -i -r --remove-all-unused-imports --ignore-init-module-imports --exclude "*/third_party/**" ./

echo "Running yapf..."
yapf -ipr -e "third_party/" -e "install/" -e "toolchain/" .

echo "Running isort..."
isort --sg "**/third_party/*" --sg "install/*" --sg "toolchain/*" ./

# Get files changed after formatting
GIT_NEW_DIFF="$(git diff --name-only)"

# Restore files that were not previously changed on this branch
for name in $GIT_NEW_DIFF; do
    if [[ $GIT_OLD_DIFF != *$name* ]]; then
        git restore $name;
    fi
done
