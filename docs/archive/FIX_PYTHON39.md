# Python 3.9 Compatibility Fix

## Issue
The code was using Python 3.10+ union type syntax (`int | None`) which is not supported in Python 3.9.

## Fix Applied
Changed all type hints to use `typing.Optional` and `typing.Union`:
- `int | None` → `Optional[int]`
- `list[Type]` → `List[Type]`

## How to Apply on ICE Cluster

After pulling the latest changes, reinstall the package:

```bash
# Navigate to the repo
cd /storage/ice-shared/cs8903onl/vector_quantization/vector-quantization

# Activate virtual environment
source .venv/bin/activate

# Reinstall in editable mode
pip install -e .

# Or force reinstall
pip install -e . --force-reinstall --no-deps
```

## Verify the Fix

```bash
# This should now work without errors:
vq-benchmark --help

# Test with a small sweep:
vq-benchmark sweep --method pq --dataset dbpedia-100k --pq-subquantizers "8" --pq-bits "8"
```

## Files Modified

- `src/haag_vq/methods/product_quantization.py` - Fixed type hints

## Python Version Requirements

- **Minimum:** Python 3.9 (now compatible)
- **Recommended:** Python 3.10+
