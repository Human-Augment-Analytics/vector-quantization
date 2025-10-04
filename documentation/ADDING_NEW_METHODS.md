# Adding New Quantization Methods - Developer Guide

This guide explains how to add new vector quantization methods to the framework and make them work with parameter sweeps.

---

## Quick Overview

To add a new quantization method, you need to:

1. Create a new quantizer class inheriting from `BaseQuantizer`
2. Add a config generator function to `sweep.py`
3. Add model creation logic to `sweep.py`
4. (Optional) Update visualization if needed

That's it! The framework handles the rest automatically.

---

## Step 1: Create Your Quantizer

All quantizers must inherit from `BaseQuantizer` and implement three methods:

```python
# src/haag_vq/methods/my_new_method.py

import numpy as np
from .base_quantizer import BaseQuantizer

class MyNewQuantizer(BaseQuantizer):
    def __init__(self, param1=default1, param2=default2):
        """
        Initialize with method-specific hyperparameters.

        Args:
            param1: Description of first parameter
            param2: Description of second parameter
        """
        self.param1 = param1
        self.param2 = param2
        # Store any state needed

    def fit(self, X: np.ndarray) -> None:
        """
        Learn quantization parameters from training data.

        Args:
            X: Training vectors, shape (N, D)
        """
        # Train your quantization model here
        # E.g., learn codebooks, compute statistics, etc.
        pass

    def compress(self, X: np.ndarray) -> np.ndarray:
        """
        Compress vectors into codes.

        Args:
            X: Vectors to compress, shape (N, D)

        Returns:
            Compressed codes (any shape/dtype that decompress can handle)
        """
        # Return compressed representation
        # E.g., cluster indices, quantized values, etc.
        pass

    def decompress(self, codes: np.ndarray) -> np.ndarray:
        """
        Decompress codes back to approximate vectors.

        Args:
            codes: Compressed codes from compress()

        Returns:
            Reconstructed vectors, shape (N, D)
        """
        # Reconstruct approximate vectors from codes
        pass

    def get_compression_ratio(self, X: np.ndarray) -> float:
        """
        Calculate compression ratio for this method.

        Args:
            X: Original vectors

        Returns:
            Compression ratio (original_bytes / compressed_bytes)
        """
        original_bytes = X.shape[1] * 4  # float32 = 4 bytes
        compressed_bytes = # calculate your method's size
        return original_bytes / compressed_bytes
```

### Example: Residual Quantization (RQ)

```python
class ResidualQuantizer(BaseQuantizer):
    def __init__(self, num_layers=4, num_clusters=256):
        self.num_layers = num_layers
        self.num_clusters = num_clusters
        self.codebooks = []

    def fit(self, X):
        residual = X.copy()
        for layer in range(self.num_layers):
            # Cluster the residuals
            kmeans = KMeans(n_clusters=self.num_clusters)
            kmeans.fit(residual)
            self.codebooks.append(kmeans.cluster_centers_)

            # Update residuals
            assigned = kmeans.cluster_centers_[kmeans.labels_]
            residual = residual - assigned

    def compress(self, X):
        codes = np.zeros((X.shape[0], self.num_layers), dtype=np.uint8)
        residual = X.copy()

        for layer in range(self.num_layers):
            dists = np.linalg.norm(
                residual[:, np.newaxis] - self.codebooks[layer],
                axis=2
            )
            codes[:, layer] = np.argmin(dists, axis=1)
            residual -= self.codebooks[layer][codes[:, layer]]

        return codes

    def decompress(self, codes):
        reconstructed = np.zeros((codes.shape[0], self.codebooks[0].shape[1]))
        for layer in range(self.num_layers):
            reconstructed += self.codebooks[layer][codes[:, layer]]
        return reconstructed

    def get_compression_ratio(self, X):
        original_bytes = X.shape[1] * 4
        compressed_bytes = self.num_layers  # 1 byte per layer
        return original_bytes / compressed_bytes
```

---

## Step 2: Add Config Generator to `sweep.py`

Add a function to generate parameter configs for your method:

```python
# In src/haag_vq/benchmarks/sweep.py

def _generate_rq_configs(layers: str, clusters: str) -> List[Dict[str, Any]]:
    """Generate Residual Quantization parameter grid."""
    configs = []
    layer_values = [int(x.strip()) for x in layers.split(",")]
    cluster_values = [int(x.strip()) for x in clusters.split(",")]

    for num_layers, num_clusters in itertools.product(layer_values, cluster_values):
        configs.append({
            "name": f"RQ(layers={num_layers}, clusters={num_clusters})",
            "num_layers": num_layers,
            "num_clusters": num_clusters
        })

    return configs
```

---

## Step 3: Wire Into Sweep Command

### 3a. Add CLI Parameters

Update the `sweep()` function signature:

```python
def sweep(
    method: str = typer.Option("pq", help="Compression method: pq, sq, rq, etc."),
    # ... existing params ...

    # Add your method's parameters
    rq_layers: str = typer.Option("2,4,8", help="[RQ only] Comma-separated layer values"),
    rq_clusters: str = typer.Option("256", help="[RQ only] Comma-separated cluster values"),

    # ... rest of params ...
):
```

### 3b. Add to Config Generation

```python
# In sweep() function, around line 70:

if method == "pq":
    configs = _generate_pq_configs(pq_chunks, pq_clusters)
elif method == "sq":
    configs = _generate_sq_configs(sq_bits)
elif method == "rq":  # ADD THIS
    configs = _generate_rq_configs(rq_layers, rq_clusters)
else:
    raise ValueError(f"Unknown method: {method}. Supported: pq, sq, rq")
```

### 3c. Add Model Creation

```python
# In _run_single_config() function, around line 164:

if method == "pq":
    model = ProductQuantizer(
        num_chunks=config["num_chunks"],
        num_clusters=config["num_clusters"]
    )
elif method == "sq":
    model = ScalarQuantizer()
elif method == "rq":  # ADD THIS
    from haag_vq.methods.residual_quantization import ResidualQuantizer
    model = ResidualQuantizer(
        num_layers=config["num_layers"],
        num_clusters=config["num_clusters"]
    )
else:
    raise ValueError(f"Unsupported method: {method}")
```

---

## Step 4: Test Your Method

```bash
# Test single run
vq-benchmark run --method rq

# Test sweep
vq-benchmark sweep --method rq \
  --rq-layers "2,4,8" \
  --rq-clusters "128,256"

# Generate plots
vq-benchmark plot
```

---

## Complete Example: Lattice Quantization

Here's a full example for a hypothetical Lattice Quantization method:

### 1. Create the quantizer

```python
# src/haag_vq/methods/lattice_quantization.py

import numpy as np
from .base_quantizer import BaseQuantizer

class LatticeQuantizer(BaseQuantizer):
    def __init__(self, lattice_dim=8, scale_factor=1.0):
        self.lattice_dim = lattice_dim
        self.scale_factor = scale_factor
        self.scaling = None

    def fit(self, X):
        # Learn optimal scaling from data
        self.scaling = np.std(X, axis=0) * self.scale_factor

    def compress(self, X):
        # Quantize to lattice points
        scaled = X / self.scaling
        quantized = np.round(scaled).astype(np.int8)
        return quantized

    def decompress(self, codes):
        return codes.astype(np.float32) * self.scaling

    def get_compression_ratio(self, X):
        original_bytes = X.shape[1] * 4  # float32
        compressed_bytes = X.shape[1]     # int8
        return original_bytes / compressed_bytes
```

### 2. Add to sweep.py

```python
# Add to imports
from haag_vq.methods.lattice_quantization import LatticeQuantizer

# Add config generator
def _generate_lattice_configs(dims: str, scales: str) -> List[Dict[str, Any]]:
    configs = []
    dim_values = [int(x.strip()) for x in dims.split(",")]
    scale_values = [float(x.strip()) for x in scales.split(",")]

    for lattice_dim, scale_factor in itertools.product(dim_values, scale_values):
        configs.append({
            "name": f"Lattice(dim={lattice_dim}, scale={scale_factor})",
            "lattice_dim": lattice_dim,
            "scale_factor": scale_factor
        })
    return configs

# Add to sweep() parameters
def sweep(
    # ... existing ...
    lattice_dims: str = typer.Option("4,8", help="[Lattice only] Lattice dimensions"),
    lattice_scales: str = typer.Option("0.5,1.0", help="[Lattice only] Scale factors"),
    # ... rest ...
):

# Add to config generation
elif method == "lattice":
    configs = _generate_lattice_configs(lattice_dims, lattice_scales)

# Add to model creation
elif method == "lattice":
    model = LatticeQuantizer(
        lattice_dim=config["lattice_dim"],
        scale_factor=config["scale_factor"]
    )
```

### 3. Use it

```bash
vq-benchmark sweep --method lattice \
  --lattice-dims "4,8,16" \
  --lattice-scales "0.5,1.0,2.0"
```

---

## Best Practices

### 1. Parameter Naming Convention
- Use method prefix: `pq_chunks`, `rq_layers`, `lattice_dims`
- Makes CLI self-documenting
- Avoids conflicts between methods

### 2. Provide Sensible Defaults
```python
rq_layers: str = typer.Option("2,4,8", help="...")
```
- Defaults should cover interesting range
- Document in help text

### 3. Config Naming
```python
{
    "name": f"RQ(layers={num_layers}, clusters={num_clusters})",
    # ... actual params ...
}
```
- Name appears in logs and plots
- Make it descriptive

### 4. Error Handling
```python
def _generate_rq_configs(...):
    configs = []
    # ... generate configs ...

    if not configs:
        raise ValueError("No valid configs generated for RQ. Check parameters.")

    return configs
```
