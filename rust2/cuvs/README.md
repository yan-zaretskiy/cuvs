# cuvs

Safe Rust bindings for [NVIDIA cuVS](https://github.com/rapidsai/cuvs) --
GPU-accelerated vector search.

cuVS contains state-of-the-art implementations of several algorithms for
running approximate nearest neighbor (ANN) and exact nearest neighbor search
on the GPU, including:

- **CAGRA** -- graph-based ANN with state-of-the-art GPU throughput
- **IVF-Flat** -- inverted-file index with uncompressed vectors
- **IVF-PQ** -- inverted-file index with product quantization
- **Brute Force** -- exact nearest neighbors
- **Vamana** -- DiskANN-compatible graph index

## Prerequisites

`cuvs` links against the cuVS C library (`libcuvs_c`) at build time through
the companion `cuvs-sys` crate. You need:

1. **libcuvs** -- the cuVS shared library, installed so that CMake's
   `find_package(cuvs)` can locate it.
2. **CUDA Toolkit** -- the CUDA runtime and headers, located via
   `find_package(CUDAToolkit)`.
3. **CMake >= 3.19** -- used by the build script for library discovery.

### Install via conda (recommended)

```bash
conda install -c rapidsai -c conda-forge libcuvs cuda-version=13
```

This installs `libcuvs`, CUDA runtime libraries, and all transitive
dependencies into your conda environment. CMake will find them automatically
through `$CONDA_PREFIX`.

### Install via pip

```bash
pip install libcuvs-cu13
```

The `cuvs-sys` build script will detect the pip-installed package by inspecting
`$VIRTUAL_ENV` or querying `python3 -c "import sysconfig; ..."`.

### (Optional) PyTorch / libtorch for the `torch` feature

The `torch` feature enables passing `tch::Tensor` directly to cuVS
operations, including tensors already resident in GPU memory. This requires a
libtorch installation that matches the `tch` crate version (currently
`tch = 0.24`, which needs PyTorch/libtorch **2.11.x**).

**Option A -- pip-installed PyTorch (easiest):**

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

Set the environment variable so `torch-sys` finds it:

```bash
export LIBTORCH_USE_PYTORCH=1
```

**Option B -- standalone libtorch download:**

Download the matching libtorch archive from
<https://pytorch.org/get-started/locally/> and point `torch-sys` at it:

```bash
export LIBTORCH=/path/to/libtorch
```

## Adding `cuvs` to your project

```toml
[dependencies]
cuvs = { version = "26.6", features = ["torch"] }
```

Available features:

| Feature | Default | Description |
|---------|---------|-------------|
| `ndarray` | no | Enables `IntoDlTensor` impls for `ndarray::ArrayRef`, allowing CPU arrays as inputs |
| `torch` | no | Enables `IntoDlTensor` impls for `tch::Tensor`, allowing GPU tensors as inputs |

At least one of `ndarray` or `torch` should be enabled for a useful workflow.
The `torch` feature is recommended for GPU-resident data, which avoids
host-to-device copies.

## Example

Build a CAGRA index on the GPU, search it, and read back the results:

```rust
use std::convert::TryFrom;

use cuvs::neighbors::cagra::{Index, IndexParams, SearchParams};
use cuvs::resources::Resources;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let res = Resources::new()?;

    // Create a random dataset on the GPU.
    let n_datapoints = 65536i64;
    let n_features = 128i64;
    let dataset = tch::Tensor::randn(
        [n_datapoints, n_features],
        (tch::Kind::Float, tch::Device::Cuda(0)),
    );

    // Build the CAGRA index.
    let params = IndexParams::try_new()?;
    let index = Index::build(&res, &params, &dataset)?;

    // Search: use the first 4 rows as queries.
    let n_queries = 4i64;
    let k = 10i64;
    let queries = dataset.narrow(0, 0, n_queries);
    let mut neighbors =
        tch::Tensor::zeros([n_queries, k], (tch::Kind::Int64, tch::Device::Cuda(0)));
    let mut distances =
        tch::Tensor::zeros([n_queries, k], (tch::Kind::Float, tch::Device::Cuda(0)));

    let search_params = SearchParams::try_new()?;
    index.search(&res, &search_params, &queries, &mut neighbors, &mut distances)?;

    // Read results back to the CPU.
    let neighbors: Vec<Vec<i64>> = Vec::try_from(&neighbors)?;
    println!("Nearest neighbors: {neighbors:?}");
    Ok(())
}
```

## Runtime library path

**libcuvs** is typically found automatically at runtime when installed via
conda (the conda environment's `lib/` directory is on the default search
path).

**libtorch** (when using the `torch` feature) may need an explicit
`LD_LIBRARY_PATH`. This is standard for `tch-rs` projects:

```bash
# If using pip-installed PyTorch:
export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# If using a standalone libtorch download:
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

## How the build scripts work

If the build fails or you hit loader errors at runtime, here is what
happens under the hood:

1. **`cuvs-sys` build script** -- runs `find_package(cuvs)` via CMake to
   locate `libcuvs_c.so` and its transitive dependencies. It emits the
   required `cargo:rustc-link-lib` and `cargo:rustc-link-search` directives.
   If CMake cannot find cuVS, the build fails with an actionable error
   message telling you which package is missing and how to install it.

2. **`cuvs` build script** -- when the `torch` feature is active, links
   `libtorch_cuda` so the CUDA dispatch backend registers its kernels. Without
   this, `tch` operations on `Device::Cuda` would fail with "operator not
   available for CUDA backend".

### Pointing CMake to a custom install

If cuVS is installed in a non-standard location, set `CMAKE_PREFIX_PATH`:

```bash
export CMAKE_PREFIX_PATH=/path/to/cuvs/install
cargo build
```

### Building `libcuvs` from source

If you build the C++ library from source (e.g., for cuVS development), point
CMake at the build tree:

```bash
export CMAKE_PREFIX_PATH=/path/to/cuvs/cpp/build
cargo build
```

## docs.rs

The crate builds on docs.rs without a GPU or CUDA installation. The `doc-only`
feature skips native library discovery in `cuvs-sys`, `tch`, and `torch-sys`,
using pre-generated FFI bindings instead.

## License

Apache-2.0. See [LICENSE](https://github.com/rapidsai/cuvs/blob/HEAD/LICENSE)
for details.
