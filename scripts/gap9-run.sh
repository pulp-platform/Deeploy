#!/bin/bash

# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# gap9-run.sh - Docker orchestration script for GAP9 SDK with USB/IP device passthrough
#
#
# This script manages the containerized GAP9 development environment, including:
# - Building the GAP9 Docker image
# - Running the usbip device manager container (for USB device passthrough)
# - Starting the GAP9 SDK container with mounted volumes
#
# Prerequisites:
#   - Docker installed and running
#   - SSH key configured for BuildKit (if building image)
#   - pyusbip server running on host (for USB passthrough)
#

set -euo pipefail

#########################################################################
# Configuration & Defaults
#########################################################################

# Image and container names
GAP9_IMAGE="ghcr.io/pulp-platform/deeploy-gap9"
USBIP_IMAGE="jonathanberi/devmgr"

DOCKER_PLATFORM="auto"
DOCKER_SHELL="/bin/zsh"

# USB/IP device settings
USBIP_HOST="host.docker.internal"
USBIP_VENDOR="15ba"
USBIP_PRODUCT="002b"

# SDK and cache directories
WORK_DIR="."
CACHE_FOLDER=".cache"

# SSH key gap9 container
SSH_PRIVATE_KEY="~/.ssh/id_ed25519"

# pyusbip configuration
PYUSBIP_REPO="https://github.com/tumayt/pyusbip"
PYUSBIP_DIR=".pyusbip"

#########################################################################
# Utility Functions
#########################################################################

# Print colored output
log_info() {
	echo -e "\033[0;36m[INFO   ]\033[0m $*"
}

log_error() {
	echo -e "\033[0;31m[ERROR  ]\033[0m $*" >&2
}

log_warn() {
	echo -e "\033[0;33m[WARN   ]\033[0m $*" >&2
}

log_success() {
	echo -e "\033[0;32m[SUCCESS]\033[0m $*"
}

# Display help message
show_help() {
	cat <<EOF
Usage: $0 <command> [options]

GAP9 Docker Orchestration Script

Commands:
  start                  Start usbip daemon and GAP9 container
  start-tmux             Start everything in a tmux session with split panes
  stop                   Stop containers
  start-usbip-host       Setup and run host-side USB/IP server (in separate terminal)
  start-gap9             Start only the GAP9 container
  start-usbip-daemon     Start only the usbip device manager container
  attach-usbip           Attach USB device to usbip daemon
  detach-usbip           Detach USB device from usbip daemon
  setup-usbip-host       One-time setup for host-side USB/IP server
  help                   Display this help message

Options:
  -i, --image NAME       Docker image name (default: $GAP9_IMAGE)
  -d, --work-dir PATH    Path to working directory (default: $WORK_DIR)
  -c, --cache-dir PATH   Cache directory (default: $CACHE_FOLDER)
  -k, --ssh-key PATH     SSH private key (default: $SSH_PRIVATE_KEY)
  -h, --host ADDR        usbip host address (default: $USBIP_HOST)
  -v, --vendor ID        USB vendor ID (default: $USBIP_VENDOR)
  -p, --product ID       USB product ID (default: $USBIP_PRODUCT)
  --platform PLATFORM    Docker platform (default: $DOCKER_PLATFORM)
  --shell SHELL          Shell to use in container (default: $DOCKER_SHELL)

Examples:
  # Start everything in tmux (recommended)
  $0 start-tmux

  # Start containers with USB device passthrough (manual terminals)
  $0 start-usbip-host    # In terminal 1
  $0 start               # In terminal 2

  # Custom working directory
  $0 -d /path/to/workdir start

  # Stop everything
  $0 stop

EOF
}

# Parse command-line arguments (collect options first, then run command)
command=""
opts=()
args=("$@")
idx=0
while [[ $idx -lt ${#args[@]} ]]; do
	case "${args[$idx]}" in
	-i | --image)
		GAP9_IMAGE="${args[$((idx + 1))]}"
		opts+=("${args[$idx]}" "${args[$((idx + 1))]}")
		idx=$((idx + 2))
		;;
	-d | --work-dir)
		WORK_DIR="${args[$((idx + 1))]}"
		opts+=("${args[$idx]}" "${args[$((idx + 1))]}")
		idx=$((idx + 2))
		;;
	-c | --cache-dir)
		CACHE_FOLDER="${args[$((idx + 1))]}"
		opts+=("${args[$idx]}" "${args[$((idx + 1))]}")
		idx=$((idx + 2))
		;;
	-k | --ssh-key)
		SSH_PRIVATE_KEY="${args[$((idx + 1))]}"
		opts+=("${args[$idx]}" "${args[$((idx + 1))]}")
		idx=$((idx + 2))
		;;
	-h | --host)
		USBIP_HOST="${args[$((idx + 1))]}"
		opts+=("${args[$idx]}" "${args[$((idx + 1))]}")
		idx=$((idx + 2))
		;;
	-v | --vendor)
		USBIP_VENDOR="${args[$((idx + 1))]}"
		opts+=("${args[$idx]}" "${args[$((idx + 1))]}")
		idx=$((idx + 2))
		;;
	-p | --product)
		USBIP_PRODUCT="${args[$((idx + 1))]}"
		opts+=("${args[$idx]}" "${args[$((idx + 1))]}")
		idx=$((idx + 2))
		;;
	--platform)
		DOCKER_PLATFORM="${args[$((idx + 1))]}"
		opts+=("${args[$idx]}" "${args[$((idx + 1))]}")
		idx=$((idx + 2))
		;;
	--shell)
		DOCKER_SHELL="${args[$((idx + 1))]}"
		opts+=("${args[$idx]}" "${args[$((idx + 1))]}")
		idx=$((idx + 2))
		;;
	help | --help)
		show_help
		exit 0
		;;
	start | start-tmux | stop | start-gap9 | start-usbip-daemon | attach-usbip | detach-usbip | stop-usbip-daemon | start-usbip-host | setup-usbip-host)
		if [[ -n "$command" ]]; then
			log_error "Multiple commands provided: $command and ${args[$idx]}"
			show_help
			exit 1
		fi
		command="${args[$idx]}"
		idx=$((idx + 1))
		;;
	*)
		log_error "Unknown option or command: ${args[$idx]}"
		show_help
		exit 1
		;;
	esac
done

# Expand path of SSH private key
SSH_PRIVATE_KEY="$(eval echo "$SSH_PRIVATE_KEY")"

## Print configuration
log_info "Configuration:"
log_info "  GAP9 Docker Image: $GAP9_IMAGE"
log_info "  Working Directory: $WORK_DIR"
log_info "  Cache Directory: $CACHE_FOLDER"
log_info "  SSH Private Key: $SSH_PRIVATE_KEY"
log_info "  USB/IP Host: $USBIP_HOST"
log_info "  USB Vendor ID: $USBIP_VENDOR"
log_info "  USB Product ID: $USBIP_PRODUCT"
log_info "  Docker Platform: $DOCKER_PLATFORM"
log_info "  Docker Shell: $DOCKER_SHELL"

#########################################################################
# usbip Host Setup Functions
#########################################################################

cmd_setup_usbip_host() {
	log_info "Setting up host-side USB/IP server..."

	# Clone pyusbip if not present
	if [ ! -d "$PYUSBIP_DIR" ]; then
		log_info "Cloning pyusbip into $PYUSBIP_DIR..."
		git clone "$PYUSBIP_REPO" "$PYUSBIP_DIR"
	else
		log_info "pyusbip directory $PYUSBIP_DIR already exists, skipping clone"
	fi

	# Create virtual environment if needed
	if [ ! -d "$PYUSBIP_DIR/.venv" ]; then
		log_info "Creating Python virtual environment..."
		python3 -m venv "$PYUSBIP_DIR/.venv"

		log_info "Installing pyusbip dependencies..."
		# shellcheck disable=SC1091
		. "$PYUSBIP_DIR/.venv/bin/activate"
		pip install --upgrade pip
		pip install libusb1
	else
		log_info "Virtual environment already exists, skipping setup"
	fi

	log_success "Host-side USB/IP setup complete"
}

cmd_start_usbip_host() {
	cmd_setup_usbip_host

	log_info "Starting host-side USB/IP server (pyusbip)..."
	log_info "This process will run in the foreground. Press Ctrl+C to stop."

	cd "$PYUSBIP_DIR" &&
		# shellcheck disable=SC1091
		. .venv/bin/activate &&
		python pyusbip.py
}

#########################################################################
# usbip Daemon Functions
#########################################################################

# Wait for usbip server to be ready
wait_for_usbip_server() {
	local max_retries=20
	local retry_count=0
	local retry_interval=1

	log_info "Waiting for pyusbip server to be ready..."

	while [ $retry_count -lt $max_retries ]; do
		if pgrep -f "python.*pyusbip\.py" >/dev/null; then
			log_success "pyusbip server is ready"
			return 0
		fi

		retry_count=$((retry_count + 1))
		log_info "  Attempt $retry_count/$max_retries: pyusbip not ready yet, retrying in ${retry_interval}s..."
		sleep "$retry_interval"
	done

	log_error "Timeout waiting for pyusbip server to start (${max_retries}s)"
	return 1
}

cmd_start_usbip_daemon() {
	# Wait for pyusbip server to be ready
	if ! wait_for_usbip_server; then
		log_error "pyusbip server did not start in time"
		log_error "Please run '$0 start-usbip-host' in a separate terminal first"
		exit 1
	fi

	# Check if container already running
	if [ -n "$(docker ps -q -f name=usbip-devmgr)" ]; then
		log_info "usbip-devmgr container already running"
		return 0
	fi

	log_info "Starting usbip-devmgr container..."
	docker run -d --rm \
		--privileged \
		--pid=host \
		--name usbip-devmgr \
		-e USBIP_HOST="$USBIP_HOST" \
		-e USBIP_VENDOR="$USBIP_VENDOR" \
		-e USBIP_PRODUCT="$USBIP_PRODUCT" \
		"$USBIP_IMAGE" \
		/bin/sh -lc 'nsenter -t1 -m sh -lc "tail -f /dev/null"'

	log_success "usbip-devmgr container started"
}

cmd_attach_usbip() {
	# First detach any existing attachment
	cmd_detach_usbip || true

	log_info "Attaching USB device to usbip-devmgr..."
	docker exec -it usbip-devmgr /bin/sh -lc 'nsenter -t1 -m sh -lc "
        usbip list -r \"$USBIP_HOST\" || { echo \"usbip list failed\"; exit 1; }
        BUSID=\$(usbip list -r \"$USBIP_HOST\" \
            | grep \"$USBIP_VENDOR:$USBIP_PRODUCT\" \
            | head -n1 \
            | cut -d\":\" -f1 \
            | xargs)
        if [ -z \"\$BUSID\" ]; then
            exit 1
        fi
        usbip attach -r \"$USBIP_HOST\" -b \"\$BUSID\"
    "'

	if [ $? -ne 0 ]; then
		log_error "Failed to attach USB device"
	else
		log_success "USB device attached successfully"
	fi
}

cmd_detach_usbip() {
	log_info "Detaching USB device..."
	docker run --rm \
		--privileged \
		--pid=host \
		-e USBIP_HOST="$USBIP_HOST" \
		-e USBIP_VENDOR="$USBIP_VENDOR" \
		-e USBIP_PRODUCT="$USBIP_PRODUCT" \
		"$USBIP_IMAGE" \
		/bin/sh -lc 'nsenter -t1 -m sh -lc "
            PORT=\$(usbip port \
                | grep \"$USBIP_VENDOR:$USBIP_PRODUCT\" -B 1 \
                | head -n1 \
                | sed -E \"s/^Port ([0-9]+):.*/\1/\" \
                | xargs)
            if [ -z \"\$PORT\" ]; then
                exit 1
            fi
            usbip detach -p \"\$PORT\"
        "' >/dev/null 2>&1

	if [ $? -ne 0 ]; then
		log_warn "Failed to detach USB device (it may not have been attached)"
	else
		log_success "USB device detached (or not attached)"
	fi
}

cmd_stop_usbip_daemon() {
	log_info "Stopping usbip-devmgr container..."

	# First detach the device
	cmd_detach_usbip || true

	# Stop the container
	if [ -n "$(docker ps -q -f name=usbip-devmgr)" ]; then
		docker stop usbip-devmgr >/dev/null 2>&1
		log_success "usbip-devmgr container stopped"
	else
		log_info "usbip-devmgr container not running"
	fi
}

#########################################################################
# GAP9 Container Functions
#########################################################################

cmd_start_gap9() {
	log_info "Starting GAP9 container..."

	# Prepare cache directory
	mkdir -p "$CACHE_FOLDER"
	touch "$CACHE_FOLDER/.zsh_history"

	# Validate WORK_DIR exists
	if [ ! -d "$WORK_DIR" ]; then
		log_error "WORK_DIR not found: $WORK_DIR"
		log_error "Use -d/--work-dir to set the SDK path"
		exit 1
	fi

	log_info "Press Ctrl+D or type 'exit' to exit container"

	# Build docker run command with optional platform argument
	local docker_run_args=(
		-it --rm
		--privileged
		-v /dev/bus/usb:/dev/bus/usb
		-v "$SSH_PRIVATE_KEY":/root/.ssh/id_ed25519:ro
		-v "$WORK_DIR/":/app/work/
		-v "$CACHE_FOLDER/.zsh_history":/root/.zsh_history
		-v "$CACHE_FOLDER/ccache":/ccache
		-e CCACHE_DIR=/ccache
	)

	# Add platform argument if not 'auto'
	if [[ "$DOCKER_PLATFORM" != "auto" ]]; then
		docker_run_args+=(--platform "$DOCKER_PLATFORM")
	fi

	docker_run_args+=("$GAP9_IMAGE" "$DOCKER_SHELL" -c "cd /app/work && $DOCKER_SHELL")

	docker run "${docker_run_args[@]}"
}

#########################################################################
# Orchestration Functions
#########################################################################

cmd_start() {
	log_info "Starting GAP9 orchestration (usbip daemon + GAP9 container)..."
	cmd_start_usbip_daemon
	cmd_attach_usbip
	cmd_start_gap9
}

cmd_stop() {
	log_info "Stopping all containers..."
	cmd_stop_usbip_daemon
	cmd_stop_tmux
	log_success "All containers stopped"
}

#########################################################################
# Tmux Orchestration
#########################################################################

cmd_start_tmux() {
	local session_name="gap9-dev"
	local script_path="$0"
	local opts_escaped=""

	for opt in "${opts[@]:-}"; do
		if [[ -n "$opt" ]]; then
			printf -v opt '%q' "$opt"
			opts_escaped+=" $opt"
		fi
	done

	# Check if tmux is installed
	if ! command -v tmux &>/dev/null; then
		log_error "tmux is not installed. Please install tmux first."
		log_error "On macOS: brew install tmux"
		log_error "On Linux: sudo apt-get install tmux"
		exit 1
	fi

	# Kill any existing session with the same name
	tmux kill-session -t "$session_name" 2>/dev/null || true

	log_info "Creating tmux session: $session_name"

	# Create new session with three panes (usbip-host, usbip-daemon, gap9)
	tmux new-session -d -s "$session_name" -x 200 -y 50

	# First pane: run pyusbip server

	# Second pane: run main orchestration (with delay to let server start)
	tmux split-window -t "$session_name:0" -h
	tmux split-window -t "$session_name:0" -v
	tmux send-keys -t "$session_name:0.1" "alias stop='$script_path$opts_escaped stop'" Enter
	tmux send-keys -t "$session_name:0.1" "$script_path$opts_escaped start-usbip-host" Enter
	tmux send-keys -t "$session_name:0.2" "alias stop='$script_path$opts_escaped stop'" Enter
	tmux send-keys -t "$session_name:0.2" "$script_path$opts_escaped start-usbip-daemon" Enter
	tmux send-keys -t "$session_name:0.2" "$script_path$opts_escaped attach-usbip" Enter
	tmux send-keys -t "$session_name:0.0" "alias stop='$script_path$opts_escaped stop'" Enter
	tmux send-keys -t "$session_name:0.0" "$script_path$opts_escaped start-gap9" Enter

	# Select the first pane
	tmux select-pane -t "$session_name:0.0"

	log_success "tmux session created: $session_name"
	log_info "Attaching to session..."
	log_info "To detach: Ctrl+B then D"
	log_info "To kill session: tmux kill-session -t $session_name"

	# Attach to the session
	tmux attach-session -t "$session_name"
}

cmd_stop_tmux() {
	local session_name="gap9-dev"
	log_info "Stopping tmux session: $session_name"
	tmux kill-session -t "$session_name" 2>/dev/null || log_info "tmux session $session_name not running"
}

#########################################################################
# Main Script Logic
#########################################################################

# If no command provided, show help
if [[ -z "$command" ]]; then
	cmd_start_tmux
	exit 0
fi

# Execute the command after options are parsed
case "$command" in
start)
	cmd_start
	;;
start-tmux)
	cmd_start_tmux
	;;
stop)
	cmd_stop
	;;
start-gap9)
	cmd_start_gap9
	;;
start-usbip-daemon)
	cmd_start_usbip_daemon
	;;
attach-usbip)
	cmd_attach_usbip
	;;
detach-usbip)
	cmd_detach_usbip
	;;
stop-usbip-daemon)
	cmd_stop_usbip_daemon
	;;
start-usbip-host)
	cmd_start_usbip_host
	;;
setup-usbip-host)
	cmd_setup_usbip_host
	;;
*)
	log_error "Unknown command: $command"
	show_help
	exit 1
	;;
esac
