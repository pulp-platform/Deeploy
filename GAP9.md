## Using Deeploy with GAP9

> ⚠️ **IMPORTANT NOTE**
> This is a work in progress. The GAP9 support in Deeploy is experimental and may not be fully functional.

To use Deeploy with GAP9, a custom Docker container is required because the official Deeploy Docker image does yet not include the necessary SDKs and dependencies for GAP9 development, because they are not publicly available.

### Build The Docker Container

To use SSH keys for accessing private repositories during the Docker build process, you need to start the SSH agent and add your SSH private key before running the Docker build command. You can do this by executing the following commands in your terminal:
```sh
cd Container
eval $(ssh-agent)

# Add your SSH private key to the agent
ssh-add ~/.ssh/id_ed25519
```

To build a local version of the Deeploy Docker image with GAP9 support using the upstream toolchain image, run:
```sh
# Build the Deeploy image with the upstream toolchain image
make deeploy DEEPOY_IMAGE=deeploy:gap9
```

Or, to build both the toolchain and Deeploy images locally, use:
```sh
# To build the toolchain container
make toolchain TOOLCHAIN_IMAGE=deeploy-toolchain:gap9 DEEPOY_IMAGE=deeploy:gap9

# To build the Deeploy container with the local toolchain image
make deeploy TOOLCHAIN_IMAGE=deeploy-toolchain:gap9 DEEPOY_IMAGE=deeploy:gap9
```

### Use The Docker Container
Then you can create and start the container in interactive mode with:

```sh
docker run -it --name deeploy_gap9 -v $(pwd):/app/Deeploy deeploy:gap9
```

Before running the test, you need to set some environment variables:
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