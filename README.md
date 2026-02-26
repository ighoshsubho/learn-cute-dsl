# learn-cute-dsl
This repo is for learning CuTe DSL and extending CUTLASS with custom TMA operations.

## TMA Gather4 / Scatter4 Support

This repo adds `tile::gather4` and `tile::scatter4` TMA mode support to CUTLASS/CuTe, which is not present in upstream CUTLASS 4.4.0.

### What is gather4/scatter4?

- **gather4**: Gathers 4 non-contiguous rows from a 2D global memory tensor into a contiguous 2D tile in shared memory
- **scatter4**: Scatters a contiguous 2D tile from shared memory into 4 non-contiguous rows in global memory
- Available on **SM100+ (Blackwell)** only, requires PTX ISA 8.6+
- The instruction takes 5 coordinates: `{col_idx, row0, row1, row2, row3}`

### Files Modified

**C++ PTX arch wrappers** (`cutlass/include/cute/arch/copy_sm100_tma.hpp`):
- `SM100_TMA_LOAD_GATHER4` — non-2SM gather4 load
- `SM100_TMA_2SM_LOAD_GATHER4` — 2SM (Blackwell 2CTA) gather4 load
- `SM100_TMA_LOAD_MULTICAST_GATHER4` — multicast gather4 load
- `SM100_TMA_2SM_LOAD_MULTICAST_GATHER4` — 2SM multicast gather4 load
- `SM100_TMA_STORE_SCATTER4` — scatter4 store

**C++ Copy Traits** (`cutlass/include/cute/atom/copy_traits_sm100_tma.hpp`):
- `Copy_Traits` specializations for all gather4/scatter4 operations
- Non-executable + executable trait pairs with `.with(mbar)` pattern

**Python CuTe DSL** (`cutlass/python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py`):
- `tma_load_gather4()` — gather4 load via `llvm.inline_asm()`
- `tma_load_gather4_multicast()` — multicast gather4 load
- `tma_store_scatter4()` — scatter4 store

### Installation

```bash
# Copy _mlir binaries from an existing nvidia-cutlass-dsl install
cp -r $(python -c "import cutlass; import os; print(os.path.dirname(cutlass.__file__))")/_mlir \
      cutlass/python/CuTeDSL/cutlass/_mlir

# Install as editable package
cd cutlass/python/CuTeDSL
pip install -e .
```

### Python DSL Usage

```python
from cutlass.cute.nvgpu.cpasync import tma_load_gather4, tma_store_scatter4

# Inside a CuTe DSL kernel:
# Gather 4 rows at indices 10, 20, 30, 40 starting at column 0
tma_load_gather4(
    tma_desc_ptr, smem_ptr, mbar_ptr,
    col_idx=Int32(0),
    row0=Int32(10), row1=Int32(20),
    row2=Int32(30), row3=Int32(40),
)

# Scatter back
tma_store_scatter4(
    tma_desc_ptr, smem_ptr,
    col_idx=Int32(0),
    row0=Int32(10), row1=Int32(20),
    row2=Int32(30), row3=Int32(40),
)
```

### C++ Usage

```cpp
#include <cute/arch/copy_sm100_tma.hpp>

// Direct PTX call
SM100_TMA_LOAD_GATHER4::copy(
    tma_desc_ptr, mbar_ptr, cache_hint, smem_ptr,
    col_idx, row0, row1, row2, row3);

// Via Copy Traits
auto tma = Copy_Traits<SM100_TMA_LOAD_GATHER4, NumBits, AuxParams>{desc, params};
auto tma_op = tma.with(mbar);
copy(tma_op, src_tensor, dst_tensor);
```
