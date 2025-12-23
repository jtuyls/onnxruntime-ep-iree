# IREE ONNX Runtime Execution Provider

ONNX Runtime Execution Provider using IREE.

## Installation

## Build

```bash
mkdir build && cd build
cmake .. -GNinja
ninja
```

Note that the plugin requires you to have iree-compiler tooling to be available
in your path. The example inference below installs it using a python env. You
can do the same.

## Example Resnet-50 Inference

We recommend using uv to manage python environments.

You can install uv from :
`https://docs.astral.sh/uv/getting-started/installation/#standalone-installer`

1. Create and activate a new virtual environment:

```bash
uv venv .env
source .env/bin/activate
```

2. Install required python packages:

```bash
uv pip install -r requirements.txt
```

3. Run jupyter notebook:

```bash
cd examples
uv run --with jupyter jupyter lab
```

4. Open resnet50.ipynb and run the cells.
