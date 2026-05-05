# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

export ZSH="$HOME/.oh-my-zsh"

# Prompt/theme selection.
ZSH_THEME="cypher"

# Keep plugin set minimal for faster shell startup.
plugins=(git)

# Load Oh My Zsh.
source $ZSH/oh-my-zsh.sh

# Quick edit shortcut for this file.
alias zshconfig="nano ~/.zshrc"

# Append and share history across sessions, without duplicates.
unsetopt HIST_SAVE_BY_COPY
setopt APPEND_HISTORY
setopt SHARE_HISTORY
setopt HIST_IGNORE_ALL_DUPS

# Ensure GAP9 SDK 'gap' command is not shadowed by an alias.
unalias gap
