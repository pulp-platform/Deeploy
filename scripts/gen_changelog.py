# ----------------------------------------------------------------------
#
# File: MchanDma.py
#
# Last edited: 11.09.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author:
# - Philip Wiese, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import subprocess
from collections import defaultdict

REPO_URL = "https://github.com/pulp-platform/Deeploy"


def get_git_commit_bodies():
    result = subprocess.run(['git', 'log', '--pretty=format:%B%n----END----'],
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE,
                            text = True)
    commits = result.stdout.strip().split("----END----")
    return [c.strip() for c in commits if c.strip()]


def get_git_commit_titles():
    result = subprocess.run(['git', 'log', '--pretty=format:%s'],
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE,
                            text = True)
    return result.stdout.split('\n')


def clean_pr_title(title):
    # Remove PR number
    title = re.sub(r'\s*\(#\d+\)', '', title)

    # Remove unwanted tags like HOTFIX, FIX, DRAFT with or without colon or brackets
    title = re.sub(r'^\s*(\[?(HOTFIX|FIX|DRAFT|OPEN|doc|HOT)\]?:?)\s*', '', title, flags = re.IGNORECASE)

    return title.strip()


def extract_pr_titles(titles):
    pr_titles = {}
    seen = set()

    for title in titles:
        pr_match = re.search(r'\(#(\d+)\)', title)
        pr_number = pr_match.group(1) if pr_match else None
        clean_title = clean_pr_title(title)

        if pr_number and clean_title not in seen:
            pr_titles[clean_title] = pr_number
            seen.add(clean_title)

    return pr_titles


def extract_list_items(commits):
    categories = defaultdict(list)

    for commit in commits:
        lines = commit.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                item = line[2:].strip()
                lowered = item.lower()

                if re.search(r'\b(add|added|introduce|initial|support|implemented)\b', lowered):
                    categories['Added'].append(item)
                elif re.search(r'\b(fix|fixed|bug)\b', lowered):
                    categories['Fixed'].append(item)
                elif re.search(r'\b(remove|removed|delete)\b', lowered):
                    categories['Removed'].append(item)
                elif re.search(
                        r'\b(change|changed|refactor|update|modify|rename|updated|improve|extend|align|adapt|bump)\b',
                        lowered):
                    categories['Changed'].append(item)
                else:
                    categories['Uncategorized'].append(item)  # fallback

    return categories


def generate_changelog(categories, pr_titles):
    lines = []

    lines.append("## Unreleased (Planned Release Target: vX.X.X)")

    # PRs list
    lines.append("### List of Pull Requests")
    for title, pr_num in pr_titles.items():
        lines.append(f"- {title} [#{pr_num}]({REPO_URL}/pull/{pr_num})")

    # Categories
    for cat in ["Added", "Changed", "Fixed", "Removed", "Uncategorized"]:
        if categories[cat]:
            lines.append(f"\n### {cat}")
            for entry in sorted(set(categories[cat])):
                lines.append(f"- {entry}")

    return '\n'.join(lines)


# Run
if __name__ == "__main__":
    commit_bodies = get_git_commit_bodies()
    commit_titles = get_git_commit_titles()

    pr_titles = extract_pr_titles(commit_titles)
    categorized_items = extract_list_items(commit_bodies)
    changelog = generate_changelog(categorized_items, pr_titles)

    with open("CHANGELOG_GEN.md", "w") as f:
        f.write(changelog)

    print("CHANGELOG_GEN.md generated.")
