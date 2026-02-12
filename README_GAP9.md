## Using Deeploy with GAP9

> ⚠️ **IMPORTANT NOTE**
> This is a work in progress. The GAP9 support in Deeploy is experimental and may not be fully functional.

To use Deeploy with GAP9, a custom Docker container is required because the official Deeploy Docker image does yet not include the necessary SDKs and dependencies for GAP9 development, because they are not publicly available.

### Build The Docker Container

To use SSH keys for accessing private repositories during the Docker build process, make sure you have an SSH key pair set up on your local machine. By default, the Makefile uses the key located at `~/.ssh/id_ed25519`. If your key is located elsewhere, you can specify its path using the `SSH_PRIVATE_KEY` variable when invoking the make command.

To build a local version of the Deeploy Docker image with GAP9 support using the upstream toolchain image, run:
```sh
cd Container

# Build the Deeploy image with the upstream toolchain image
make deeploy-gap9 DEEPOY_GAP9_IMAGE=deeploy-gap9:latest

# If you want to specify a custom SSH key path, use:
make deeploy-gap9 DEEPOY_GAP9_IMAGE=deeploy-gap9:latest SSH_PRIVATE_KEY=/path/to/your/private/key
```

Or, to build the toolchain, Deeploy and GAP9 images locally, use:
```sh
cd Container

# To build the Deeploy container with the local toolchain image
make deeploy TOOLCHAIN_IMAGE=deeploy-toolchain:gap9 DEEPOY_IMAGE=deeploy:gap9

# To build the Deeploy GAP9 container with the local toolchain image
make deeploy-gap9 TOOLCHAIN_IMAGE=deeploy-toolchain:gap9 DEEPOY_IMAGE=deeploy:gap9 DEEPOY_GAP9_IMAGE=deeploy-gap9:latest
```

### Use The Docker Container

Once the image is built, you can create and start the container in interactive mode with:

```sh
docker run -it --name deeploy_gap9 -v $(pwd):/app/Deeploy deeploy-gap9:latest
```

Before running tests, you need to set up the GAP9 environment inside the container:
```sh
source /app/install/gap9-sdk/.gap9-venv/bin/activate
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
```
Install Deeploy inside the container in editable mode:

```sh
cd /app/Deeploy
pip install -e . --extra-index-url=https://pypi.ngc.nvidia.com
```

```sh
cd /app/Deeploy/DeeployTest
python deeployRunner_gap9.py -t ./Tests/Kernels/FP32/MatMul
python deeployRunner_tiled_gap9.py -t ./Tests/Kernels/FP32/MatMul
```

### Use A Real GAP9 Board (USB/IP via gap9-run.sh)

For board access, use the orchestration script in [scripts/gap9-run.sh](scripts/gap9-run.sh). It manages:
- host-side USB/IP server (pyusbip)
- usbip device manager container
- GAP9 SDK container with the correct mounts

#### Prerequisites
- Docker installed and running
- A working SSH key for BuildKit (if you are building the image locally)
- USB/IP host support (the script can set up pyusbip on the host)


#### Start the board workflow (recommended)
This launches a tmux session with two panes: one for the host USB/IP server and one for the GAP9 container.

```sh
./scripts/gap9-run.sh start-tmux
```

#### Start manually (two terminals)
Terminal 1 (host USB/IP server):
```sh
./scripts/gap9-run.sh start-usbip-host
```

Terminal 2 (containers + attach device):
```sh
./scripts/gap9-run.sh start
```

#### Common options and environment variables
- Use a custom image: `-i your-gap9-image:tag` or `GAP9_IMAGE=your-gap9-image:tag`
- Set USB device IDs: `USBIP_VENDOR=15ba USBIP_PRODUCT=002b`
- Change USB/IP host: `USBIP_HOST=host.docker.internal`

To stop everything:
```sh
./scripts/gap9-run.sh stop
```
