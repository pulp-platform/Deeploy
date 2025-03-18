#!/bin/bash

# Define directories to exclude
EXCLUDE_DIRS="third_party/ install/ toolchain/ DeeployTest/Tests"
EXCLUDE_FLAGS=()

# Build yapf exclude flags
for dir in $EXCLUDE_DIRS; do
    EXCLUDE_FLAGS+=("-e" "$dir")
done

# Initialize log file
echo "Starting code formatting..." > formatting_report.log

# Run yapf and automatically fix formatting issues
echo "Running yapf to fix Python code formatting..." >> formatting_report.log
yapf -rpi "${EXCLUDE_FLAGS[@]}" .
yapf_exit_code=$?
if [ $yapf_exit_code -ne 0 ]; then
    echo "Error occurred during yapf fixes. Exit code: $yapf_exit_code" >> formatting_report.log
    echo "Error occurred during yapf fixes. See formatting_report.log for details."
else
    echo "yapf formatting completed!" >> formatting_report.log
fi

# Run isort and automatically fix import ordering
echo "Running isort to fix Python import ordering..." >> formatting_report.log
isort --sg "**/third_party/*" --sg "install/*" --sg "toolchain/*" ./
isort_exit_code=$?
if [ $isort_exit_code -ne 0 ]; then
    echo "Error occurred during isort fixes. Exit code: $isort_exit_code" >> formatting_report.log
    echo "Error occurred during isort fixes. See formatting_report.log for details."
else
    echo "isort formatting completed!" >> formatting_report.log
fi

# Run autoflake and automatically remove unused imports
echo "Running autoflake to remove unused imports..." >> formatting_report.log
autoflake -r --in-place --remove-all-unused-imports --ignore-init-module-imports --exclude "*/third_party/**" ./
autoflake_exit_code=$?
if [ $autoflake_exit_code -ne 0 ]; then
    echo "Error occurred during autoflake fixes. Exit code: $autoflake_exit_code" >> formatting_report.log
    echo "Error occurred during autoflake fixes. See formatting_report.log for details."
else
    echo "autoflake cleanup completed!" >> formatting_report.log
fi

echo "All formatting operations completed. See formatting_report.log for details."