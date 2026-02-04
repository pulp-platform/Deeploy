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

# Build all images
make all TOOLCHAIN_IMAGE=deeploy-toolchain:latest DEEPOY_IMAGE=deeploy:latest DEEPOY_GAP9_IMAGE=deeploy-gap9:latest
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
python testRunner_gap9.py -t Tests/testMatMul
python testRunner_tiled_gap9.py -t Tests/testMatMul
```