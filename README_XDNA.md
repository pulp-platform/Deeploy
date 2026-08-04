# How to use Deeploy on the XDNA2 NPU

A dockerfile containing everything required to run on XDNA2 is available to build with the dockerfile at `Container/Dockerfile.deeploy-xdna`.

You can build it locally on Ubuntu 24.04 with:
```
docker build -f Container/Dockerfile.deeploy-xdna -t deeploy-xdna:local .
```

You need to have XRT installed on your host, once installed it is present in `/opt/xilinx/xrt`. You can run the docker container previously built with:
```
docker run -it \
  --device /dev/accel/accel0 \
  --ulimit memlock=-1 \
  -v "$(pwd)":/app/Deeploy \
  -v /opt/xilinx:/opt/xilinx \
  --name deeploy_dev \
  deeploy-xdna:local
```

Currently I use the IRON repo to generate my MLIR code, hence I have `-v /scratch/jungvi/IRON:/opt/IRON`, and `-e IRON_OPERATORS_DIR=/opt/IRON/iron/operators`. This will be as soon as the midend and backend of Deeploy are updated to support true MLIR generation.

Once the container is started you can run a simple Add node, from ONNX to execution with:
```
pip install -e ./ && \
cd DeeployTest && \
python deeployRunner_xdna2.py -t ./Tests/Kernels/BF16/Add/Regular/
```

## CI with a Self-Hosted Runner

XDNA2 tests run on a self-hosted GitHub Actions runner with NPU access.
The Docker image is built locally on the runner (not distributed via GHCR).

### One-time setup on the runner machine

1. Build the Docker image:
   ```
   docker build -f Container/Dockerfile.deeploy-xdna -t deeploy-xdna:local .
   ```

2. Register the GitHub Actions runner (Settings → Actions → Runners → New self-hosted runner).
   Use the label **`xdna2-npu`** and install as a service:
   ```
   ./svc.sh install && ./svc.sh start
   ```

3. Make sure the runner user has access to `/dev/accel/accel0` (e.g. is in the `render` group).

Once the runner is registered, pushes and PRs automatically trigger the
`CI • XDNA2` workflow defined in `.github/workflows/ci-platform-xdna2.yml`.