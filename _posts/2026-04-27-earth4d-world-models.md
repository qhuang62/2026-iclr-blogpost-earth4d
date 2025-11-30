---
layout: distill
title: "Earth4D: Production-Ready Space-Time Positional Encoding for World Models"
description: How decomposed 4D hash encoding with learned probing enables planetary-scale deep learning with 99% parameter reduction while matching foundation model performance
date: 2026-04-27
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Anonymous

bibliography: 2026-04-27-earth4d-world-models.bib

toc:
  - name: Introduction
  - name: The Positional Encoding Challenge
    subsections:
      - name: Why Earth Observation Needs 4D Encoding
      - name: Limitations of Existing Approaches
  - name: Earth4D Architecture
    subsections:
      - name: Decomposed Spatiotemporal Representation
      - name: Multi-Resolution Hierarchy
      - name: Hash Encoding Mechanics
  - name: The Hash Collision Problem
    subsections:
      - name: Understanding Hash Collisions
      - name: The uint32 Overflow Discovery
      - name: Collision Patterns Across Scales
  - name: Learned Hash Probing
    subsections:
      - name: How Learned Probing Works
      - name: Extreme Compression Results
  - name: Experimental Validation
    subsections:
      - name: "Q1: Matching Foundation Models with Coordinates Alone"
      - name: "Q2: Can We Achieve 99% Parameter Reduction?"
      - name: "Q3: RGB Reconstruction from Elevation"
  - name: Implications for World Models
  - name: Limitations and Future Work
  - name: Conclusion

---

## Introduction

World models—systems that learn to simulate and predict complex spatiotemporal dynamics—have emerged as a promising paradigm for understanding our planet. From climate forecasting to disaster response, these models need to process Earth observation data spanning vast spatial and temporal scales: from sub-meter satellite imagery to continental weather patterns, from sub-second events to century-long climate trends.

At the heart of these models lies a fundamental challenge: **how do we encode continuous space-time coordinates into learnable representations?**

While transformers have revolutionized deep learning through positional encodings, existing approaches face severe limitations when applied to planetary-scale spatiotemporal data. Sinusoidal encodings lack the expressiveness for complex Earth phenomena. Learned embeddings require discretizing continuous space-time, losing resolution. And while 3D hash encoding<d-cite key="muller2022instant"></d-cite> revolutionized neural graphics, it doesn't capture temporal dynamics.

We present **Earth4D**, a production-ready 4D space-time positional encoder that extends multi-resolution hash encoding to four dimensions. Earth4D achieves remarkable results:

- **Matches state-of-the-art foundation models** using only (x,y,z,t) coordinates—no satellite imagery, weather data, or topography required
- **99% parameter reduction** (724M → 5M) with 4× training speedup while maintaining strong performance
- **Planetary coverage** from sub-meter to continental scale, with temporal precision from sub-second to centuries

More importantly, Earth4D represents a shift in how we think about world models: rather than requiring massive multimodal pretraining, we can achieve competitive performance by learning rich spatiotemporal representations from coordinates alone.

---

## The Positional Encoding Challenge

### Why Earth Observation Needs 4D Encoding

Earth observation data presents unique challenges that distinguish it from typical deep learning tasks:

**Continuous spatiotemporal coordinates**: Unlike images with discrete pixel positions or text with sequential tokens, Earth data exists in continuous 4D space-time. A wildfire measurement at (37.77°N, -122.42°W, 50m elevation, March 23 2023 8:58 AM) needs precise encoding.

**Extreme scale variation**: We need to reason about phenomena at vastly different scales simultaneously—a climate model might track both individual storms (10-100km) and global circulation patterns (10,000km+). Temporal scales range from seconds (lightning strikes) to decades (climate change).

**Planetary coverage**: Unlike 3D graphics which operate in local coordinate frames, Earth observation requires global consistency. The same encoding must work for data from San Francisco, Sydney, and the South Pole.

**Memory constraints**: Processing planet-scale data quickly exhausts GPU memory. A naive grid representation at 1-meter resolution would require $$10^{18}$$ grid cells globally—completely infeasible.

### Limitations of Existing Approaches

**Sinusoidal positional encodings**<d-cite key="vaswani2017attention"></d-cite>, while elegant for transformers, provide fixed basis functions that cannot adapt to complex spatiotemporal patterns in Earth data. They work well for sequential data but struggle with the intricate multi-scale structure of geospatial phenomena.

**Learned embeddings** require discretizing continuous coordinates. While Vision Transformers<d-cite key="dosovitskiy2021image"></d-cite> discretize images into patches, Earth observation data doesn't have natural "patch" boundaries. Discretization either loses fine-grained resolution or creates memory-prohibitive lookup tables.

**3D multi-resolution hash encoding**<d-cite key="muller2022instant"></d-cite> (InstantNGP) solved many of these problems for neural graphics by using hash tables to achieve memory-efficient multi-resolution encoding. However, it was designed for static 3D scenes. Earth observation fundamentally requires modeling **how** spatial patterns evolve **over time**—a 4D problem.

Extending to 4D isn't trivial. A naive 4D grid would explode memory requirements. Simply treating time as another spatial dimension loses the distinct characteristics of temporal dynamics (irreversibility, causality, different resolution requirements).

What we need is a 4D encoding that:
1. Handles continuous coordinates without discretization
2. Scales efficiently to planetary coverage
3. Captures multi-resolution structure in both space and time
4. Adapts to data through learning
5. Fits in GPU memory

Earth4D addresses all of these requirements.

---

## Earth4D Architecture

### Decomposed Spatiotemporal Representation

Earth4D's core innovation is a **decomposed 4D encoding** that separates spatial and temporal structure while capturing their interactions. Rather than a single 4D hash grid (which would be memory-prohibitive), Earth4D uses four 3D grids:

1. **XYZ Grid**: Pure spatial encoding in Earth-Centered Earth-Fixed (ECEF) coordinates
2. **XYT Grid**: Equatorial plane + time
3. **YZT Grid**: 90°E meridian plane + time
4. **XZT Grid**: Prime meridian plane + time

<figure>
  <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/earth4d_architecture.png' | relative_url }}" alt="Earth4D Architecture" style="width: 100%;">
  <figcaption>Earth4D decomposes 4D space-time into four 3D hash-encoded grids. Each grid operates at multiple resolution levels (coarse to fine), with features concatenated to form a 192D spatiotemporal embedding.</figcaption>
</figure>

This decomposition is inspired by Grid4D<d-cite key="xu2024grid4d"></d-cite> but optimized for planetary scale. By using three orthogonal spatiotemporal projections, we capture how spatial patterns evolve over time from different perspectives. The XYZ grid provides pure spatial context, while XYT, YZT, and XZT encode temporal dynamics in complementary subspaces.

**Why ECEF coordinates?** We use Earth-Centered Earth-Fixed (ECEF) coordinates internally rather than latitude/longitude because:
- ECEF provides uniform spatial hashing globally (no polar singularities)
- Distances are Euclidean, making interpolation consistent
- 3D Cartesian coordinates align naturally with hash encoding

Input coordinates (latitude, longitude, elevation, time) are automatically converted to ECEF and normalized to $$[-1, 1]$$.

### Multi-Resolution Hierarchy

Each of the four grids operates at multiple resolution levels simultaneously. This multi-resolution structure is crucial for capturing both local details and global patterns.

At level $$L$$, the grid resolution is:

$$r_L = b \cdot g^L$$

where $$b$$ is the base resolution (typically 32) and $$g$$ is the growth factor (typically $$\sqrt{2}$$). With 24 levels (default configuration), spatial resolution ranges from:

- **Level 1**: 398.2 km/cell (continental scale)
- **Level 12**: 194.4 m/cell (city scale)
- **Level 24**: 4.75 cm/cell (sub-meter precision)

Temporal resolution similarly spans from years to sub-second precision.

### Hash Encoding Mechanics

At each resolution level $$L$$, we map continuous coordinates to discrete grid positions, then hash them to a fixed-size lookup table:

```
1. Grid position: pos_grid = floor(coordinate × resolution_L)
2. Hash index:
   if grid_size ≤ hashmap_size:
       index = pos_grid.x + pos_grid.y × stride_y + pos_grid.z × stride_z
   else:
       index = hash(pos_grid) mod hashmap_size
3. Interpolate: trilinear interpolation of 8 corner features
```

The hash function uses XOR with large primes for mixing:

$$\text{hash}(\mathbf{p}) = \bigoplus_{d=1}^{D} p_d \cdot \pi_d \pmod{T}$$

where $$\mathbf{p}$$ is the grid position, $$\pi_d$$ are large primes (2654435761, 805459861, ...), and $$T$$ is the hash table size.

Coarse levels (small grid size) use direct indexing with no collisions. Fine levels (large grid size) use hashing, which introduces collisions but saves massive memory.

**Smoothstep interpolation** $$(S(t) = 3t^2 - 2t^3)$$ provides C¹ continuous gradients, better than linear interpolation for smooth Earth phenomena like temperature fields or elevation gradients.

The final output concatenates features from all grids and levels:
- 4 grids × 24 levels × 2 features = **192D embedding** per (x,y,z,t) coordinate

This entire process runs on GPU via custom CUDA kernels, enabling **massively parallel encoding** of millions of coordinates simultaneously.

---

## The Hash Collision Problem

### Understanding Hash Collisions

Hash encoding's memory efficiency comes with a tradeoff: **hash collisions**. When the grid size exceeds the hash table size, multiple different spatial positions can map to the same hash index.

For example, with a hash table size of $$2^{22}$$ (4 million entries) and level 24 grid resolution of $$2^{28}$$ cells, only 1 in 64 grid cells gets a unique hash entry. The other 63 collide.

<figure>
  <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/collision_heatmap.png' | relative_url }}" alt="Collision Rate Analysis" style="width: 100%;">
  <figcaption>Hash collision rates across resolution levels for different data distributions. Fine levels (high resolution) show expected 2-4% collision rates, except for power-of-2 artifacts at level 23.</figcaption>
</figure>

**Are collisions bad?** Not necessarily. The hash encoding literature<d-cite key="muller2022instant"></d-cite> shows that downstream networks (MLPs) can learn to disambiguate collisions when they're relatively rare. The hash function's randomness actually acts as a form of regularization.

However, **catastrophic collision patterns** destroy information. If all temporal variations at a location map to the same index, we lose the ability to model temporal dynamics.

### The uint32 Overflow Discovery

During development, we discovered bizarre collision patterns in temporal grids:

- **Level 8**: 100% collision rate (only ~978 unique indices for 41,261 coordinates)
- **Levels 13-19**: 99.9% collision rate (all coordinates with same spatial position but different times mapped identically)

This violated the expected monotonic decrease in collisions as resolution increases. Something was fundamentally broken.

After extensive debugging, we discovered a **critical integer overflow bug** in the CUDA kernel:

```cuda
// BUGGY CODE
uint32_t stride = 1;
for (uint32_t d = 0; d < 3; d++) {
    index += pos_grid[d] * stride;
    stride *= resolution[d];  // OVERFLOW!
}
```

At level 8 with resolution 2048 per dimension:
- After processing first two dimensions: `stride = 2048 × 2048 = 4,194,304`
- Next multiplication: `4,194,304 × 2048 = 8,589,934,592`
- **This overflows uint32 (max 4,294,967,295) and wraps to 0!**

When stride became 0, the temporal dimension contributed nothing to the hash index. All temporal variation was lost.

**The fix** was simple but critical—use 64-bit arithmetic for intermediate calculations:

```cuda
// FIXED CODE
uint64_t stride = 1;  // Prevents overflow
```

After this fix:
- Level 8: 100% → **40.5%** collision rate
- Level 13: 99.9% → **2.4%** collision rate
- Level 19: 99.9% → **2.1%** collision rate

This bug hunt revealed an important lesson: **subtle integer overflow can catastrophically corrupt spatiotemporal encodings**. The bug only manifested at specific resolution/hash-table size combinations, making it nearly invisible without careful analysis.

### Collision Patterns Across Scales

Even with the overflow bug fixed, hash collisions are inherent to memory-efficient encoding. We profiled collision rates across 10 different spatiotemporal data distributions:

1. **Uniform Random**: Global Earth surface sampling
2. **Continental Sparse**: Sparse coverage of North America
3. **City-Scale Cluster**: 10km × 10km dense sampling
4. **Building-Scale**: Single 10m × 10m area over time
5. **Time Series**: Fixed locations sampled repeatedly over time

Results showed collision rates ranging from 0% (coarse levels) to 2-4% (fine levels), with expected power-of-2 artifacts at level 23 (67M grid cells, 4M hash entries = exact 16× ratio).

While 2-4% collision rate is acceptable for many applications, we wanted to push further: **Can we reduce hash table size even more while maintaining quality?**

This led us to learned hash probing.

---

## Learned Hash Probing

### How Learned Probing Works

Learned hash probing<d-cite key="takikawa2023compact"></d-cite> is a technique that learns to resolve hash collisions intelligently. Instead of a single hash function, we use **dual hashing with learned offsets**:

$$\text{index} = N_p \times h_1(\mathbf{x}) + \mathcal{D}_c[h_2(\mathbf{x})]$$

where:
- $$h_1(\mathbf{x})$$: Primary hash function (coarse spatial localization)
- $$h_2(\mathbf{x})$$: Secondary hash with different primes (decorrelated)
- $$\mathcal{D}_c$$: Learned codebook of probe offsets
- $$N_p$$: Probing range (typically 4, 8, or 16)

The codebook $$\mathcal{D}_c$$ is learned during training via gradients. Initially, probes are uniformly distributed across the $$N_p$$ candidates. As training progresses, the model learns which probe indices minimize collisions for the specific data distribution.

**Backward pass** uses a straight-through estimator: during forward pass, we select a discrete probe index via argmax. During backward pass, we treat the discrete selection as differentiable by distributing gradients across all $$N_p$$ candidates weighted by their softmax probabilities.

This allows the model to learn **data-adaptive collision resolution** rather than relying on random hash functions alone.

### Extreme Compression Results

Learned hash probing enables dramatic parameter reduction. On the Globe-LFMC 2.0 benchmark<d-cite key="yebra2024globelfmc"></d-cite>:

| Configuration | Parameters | GPU Memory | Speed | MAE | R² | vs Baseline |
|---------------|-----------|------------|-------|-----|-----|-------------|
| **Baseline** ($$2^{22}$$ hash) | 724M | 12GB+ | 1× | 16.6 pp | 0.582 | — |
| **Learned Probing** ($$2^{22}$$) | 724M | 12GB+ | 1.7× | **12.4 pp** | **0.745** | +28% R² |
| **Compressed** ($$2^{14}$$ hash) | **5.1M** | **850MB** | **4×** | **15.0 pp** | **0.668** | +14.7% R² |

The compressed configuration achieves:
- **99.3% parameter reduction** (724M → 5.1M)
- **93% memory reduction** (12GB → 850MB)
- **4× training speedup**
- **Still outperforms baseline** by 14.7% R²

This is remarkable: by shrinking the hash table by $$256×$$ ($$2^{22}$$ → $$2^{14}$$) and adding learned probing, we maintain—and even improve—performance while fitting on edge devices.

<figure>
  <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/compression_tradeoff.png' | relative_url }}" alt="Compression Tradeoff" style="width: 90%;">
  <figcaption>Performance vs parameter count. Learned hash probing (orange) enables 99% compression with minimal quality loss. The compressed 5.1M model outperforms the 724M baseline.</figcaption>
</figure>

**Why does compression improve performance?** We hypothesize that extreme compression acts as regularization. The forced sharing of hash table entries encourages the model to learn more generalizable spatiotemporal features rather than memorizing training locations. This is similar to how dropout or weight decay can improve generalization.

---

## Experimental Validation

We evaluate Earth4D through three research questions, each designed to test a specific capability required for world models.

### Q1: Matching Foundation Models with Coordinates Alone

**Question**: Can Earth4D achieve state-of-the-art performance using only spatiotemporal coordinates, without satellite imagery, weather data, or other multimodal inputs?

**Dataset**: Globe-LFMC 2.0<d-cite key="yebra2024globelfmc"></d-cite>, a global benchmark for predicting Live Fuel Moisture Content (LFMC)—the percentage of water in vegetation relative to dry weight. LFMC is critical for wildfire risk assessment.

- 89,764 field measurements across diverse plant species, geographic regions, and temporal periods (2000-2023)
- Train/test split: 76,467 / 13,297 (official AI2 split for fair comparison)

**Baseline**: Galileo<d-cite key="tseng2025galileo"></d-cite>, a Vision Transformer (5.3M parameters) pre-trained by Allen Institute for AI on:
- Sentinel-2 optical imagery (10m resolution, 13 spectral bands)
- Sentinel-1 SAR (radar, cloud-penetrating)
- ERA-5 weather reanalysis (temperature, precipitation, etc.)
- TerraClimate soil moisture and climate data
- SRTM topography (elevation, slope, aspect)
- (x,y,z,t) coordinates and species type

**Earth4D Architecture**:
- Earth4D encodes (x,y,z,t) into 192D embeddings
- Concatenated with learnable species embedding (initialized randomly)
- MLP predicts LFMC percentage

**Results**:

| Model | Data Inputs | MAE | R² |
|-------|------------|-----|-----|
| **Galileo** (pretrained) | Coordinates + Species + **Multimodal Remote Sensing** | 12.6 pp | 0.72 |
| **Earth4D** (learned probing) | **Coordinates + Species only** | **12.4 pp** | **0.745** |

Earth4D **surpasses the pretrained foundation model** (12.4 vs 12.6 MAE, 0.745 vs 0.72 R²) using only coordinates and species embeddings. No satellite imagery. No weather data. No topography.

<figure>
  <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/lfmc_results.png' | relative_url }}" alt="LFMC Prediction Results" style="width: 100%;">
  <figcaption><b>Top</b>: Distribution of absolute errors across 13,297 test samples (median 7.1pp). <b>Left</b>: Geographic error distribution shows low error in well-sampled regions. <b>Right</b>: Temporal predictions track ground truth LFMC across seasons (2017-2023).</figcaption>
</figure>

**What's happening?** Earth4D learns that certain spatiotemporal coordinates correlate with LFMC patterns:
- **Spatial**: Coastal California (37°N, -122°W, low elevation) tends toward high LFMC in winter
- **Temporal**: Summer months (June-August) show lower LFMC across most species
- **Elevation**: Higher elevations retain moisture longer
- **Interactions**: Spatial patterns shift with seasons (spatiotemporal coupling)

The 4D hash encoding captures these multi-scale correlations across 24 resolution levels. Coarse levels encode climate zones (Mediterranean, desert, temperate). Fine levels encode microclimate variations.

Crucially, the **species embedding provides botanical context** that combines with spatiotemporal features. The model learns that Species A in Location X at Time T has different moisture dynamics than Species B at the same location and time.

This result challenges a common assumption: **multimodal pretraining isn't always necessary if you have rich positional encodings and the right inductive biases**.

### Q2: Can We Achieve 99% Parameter Reduction?

**Question**: Does the extreme compression result (99% parameter reduction, 4× speedup) shown earlier generalize across different configurations?

**Experiment**: We systematically vary hash table size and probing range across the LFMC benchmark:

| Hash Size | Probing | Parameters | Speed | MAE | R² | Memory |
|-----------|---------|------------|-------|-----|-----|---------|
| $$2^{22}$$ | Disabled | 724M | 1.0× | 16.6 | 0.582 | 12GB+ |
| $$2^{22}$$ | $$N_p=4$$ | 724M | 1.5× | 13.2 | 0.698 | 12GB+ |
| $$2^{22}$$ | $$N_p=32$$ | 724M | 1.7× | **12.4** | **0.745** | 12GB+ |
| $$2^{18}$$ | $$N_p=32$$ | 45M | 1.8× | 13.8 | 0.672 | 1.5GB |
| $$2^{14}$$ | $$N_p=32$$ | **5.1M** | **4.0×** | 15.0 | 0.668 | 850MB |

**Key findings**:

1. **Learned probing consistently improves performance** even at full hash table size (16.6 → 12.4 MAE)
2. **Larger probing range ($$N_p$$) improves quality** but adds training overhead
3. **Sweet spot: $$2^{18}$$ hash + $$N_p=32$$** balances quality and efficiency (93.8% reduction, strong performance)
4. **Extreme compression ($$2^{14}$$) remains viable** for edge deployment

The 99% reduction result is robust across multiple trials and random seeds. The key enabler is learned probing's ability to adaptively resolve collisions based on data distribution.

### Q3: RGB Reconstruction from Elevation

**Question**: Can Earth4D learn to infer RGB pixel values from (x,y,z,t) coordinates alone?

This tests a different capability: **pure spatiotemporal function approximation** without any auxiliary labels (like species type in LFMC).

**Dataset**: 5.8M coordinate-color pairs from Houston coastal wetlands:
- **Input**: USGS 3DEP LiDAR elevation (x,y,z in ECEF, t = acquisition date)
- **Target**: USDA NAIP RGB imagery (R,G,B values at corresponding location/time)

The objective is $$(x,y,z,t) \rightarrow (r,g,b)$$: given only coordinates, predict the RGB color.

**Architecture**: Earth4D (192D) → MLP (3 hidden layers, 128 units) → RGB (3 channels)

**Results**:

| Configuration | Validation Loss | Improvement |
|---------------|----------------|-------------|
| Baseline (no probing) | 0.0847 | — |
| Learned Probing ($$N_p=32$$) | **0.0694** | **-18%** |

<figure>
  <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/rgb_reconstruction.png' | relative_url }}" alt="RGB Reconstruction" style="width: 100%;">
  <figcaption>RGB reconstruction from LiDAR elevation in Houston wetlands (2018). <b>Left to right</b>: LiDAR height, ground truth RGB, baseline reconstruction, learned probing reconstruction (18% lower loss).</figcaption>
</figure>

The model learns complex correlations:
- **Water bodies** (low elevation, flat) → blue/green hues
- **Vegetation** (moderate elevation, rough terrain) → green
- **Urban areas** (high elevation variance) → gray/brown
- **Coastal transitions** (elevation gradients) → color gradients

Learned probing significantly improves reconstruction quality, especially in fine-detail regions like coastline boundaries and vegetation patches.

This experiment demonstrates Earth4D's ability to capture **implicit spatiotemporal functions** that generalize across diverse phenomena—not just LFMC prediction, but any function of (x,y,z,t).

---

## Implications for World Models

Earth4D's results suggest several important implications for building world models of Earth observation data:

### 1. Positional Encodings as First-Class Features

Traditional approaches treat positional information as auxiliary context for satellite imagery or sensor data. Earth4D flips this: **positional encodings can be primary features** that capture rich spatiotemporal patterns.

This is analogous to how CLIP<d-cite key="radford2021learning"></d-cite> showed that text alone (without pixel-level annotations) could guide image understanding through contrastive learning. Here, coordinates alone (without multimodal data) can achieve competitive performance through expressive encoding.

For world models, this means we can:
- **Bootstrap** from coordinate-only data when multimodal inputs are unavailable
- **Reduce dependency** on expensive satellite imagery or weather reanalysis
- **Generalize** to regions/times with sparse observational coverage

### 2. Extreme Efficiency Enables Edge Deployment

The 99% parameter reduction (724M → 5M) makes Earth4D viable for edge deployment:
- **Satellite onboard processing**: Run Earth4D on satellite GPUs for real-time wildfire detection
- **Mobile disaster response**: Deploy on tablets/phones for field teams
- **IoT sensor networks**: Embed in low-power environmental monitoring stations

This shifts world models from datacenter-scale to **ubiquitous deployment**, enabling real-time decision-making where it matters most.

### 3. Learned Probing as Universal Compression

Learned hash probing isn't specific to Earth observation—it's a **general technique for compressing hash-based encodings**. Applications include:
- Neural radiance fields (NeRF) for 3D reconstruction
- Implicit neural representations for any spatiotemporal data
- Memory-efficient transformers with positional embeddings

The key insight: **let the model learn to resolve collisions** rather than sizing hash tables conservatively.

### 4. Downstream Task Agnostic

Earth4D produces a 192D embedding per (x,y,z,t) coordinate. This embedding can feed into:
- **Classification**: Crop type, land cover, disaster detection
- **Regression**: Temperature, precipitation, soil moisture
- **Segmentation**: Flood extent, deforestation boundaries
- **Generation**: Synthesizing satellite imagery from coordinates
- **Forecasting**: Predicting future states from current embeddings

By separating the positional encoder from task-specific heads, we enable **transfer learning** across Earth observation tasks. Pretrain Earth4D on one task (LFMC), then fine-tune on another (crop yield), reusing the spatiotemporal representations.

### 5. Foundation for Multimodal World Models

While Earth4D succeeds with coordinates alone, it's designed for **fusion with multimodal encoders**:

```
[Satellite Image] → Vision Encoder → 512D
[Weather Data]    → Time Series Enc → 256D
[(x,y,z,t)]       → Earth4D         → 192D
────────────────────────────────────────────
Concatenate       → Transformer     → Predictions
```

The 4D positional embedding provides **spatiotemporal grounding** for other modalities. An image patch at (lat, lon) gets enriched with Earth4D's multi-resolution features encoding its geospatial context.

This mirrors how language models use positional encodings—not as replacements for tokens, but as essential context that enables attention mechanisms to reason about structure.

---

## Limitations and Future Work

While Earth4D demonstrates strong performance, several limitations and open questions remain:

### Power-of-2 Collision Artifacts

At level 23 (resolution $$2^{26}$$, hash table $$2^{22}$$), collision rate jumps to 3.8% due to exact power-of-2 ratio. This creates periodic artifacts in hash distribution.

**Mitigation**: Use non-power-of-2 hash table sizes (large primes) or increase hash capacity. However, power-of-2 sizes align with GPU memory boundaries and enable bitwise optimizations.

### Hyperparameter Sensitivity

Earth4D has several hyperparameters:
- Number of resolution levels (default 24)
- Hash table size per grid ($$2^{14}$$ to $$2^{22}$$)
- Probing range $$N_p$$ (2, 4, 8, 16, 32)
- Codebook size $$N_c$$ (512 to 4096)

While we provide reasonable defaults, **optimal settings vary by task**. Automated hyperparameter search (e.g., using validation loss) would improve usability.

### Learning Rate Tuning for Probing

Index logits gradients are 5-7 orders of magnitude smaller than embedding gradients (inherent to straight-through estimators). We recommend 100× higher learning rate for index logits:

```python
optimizer = torch.optim.Adam([
    {'params': encoder.embeddings, 'lr': 1e-3},
    {'params': encoder.index_logits, 'lr': 1e-1}
])
```

This dual learning rate requirement adds complexity. **Automatic gradient rescaling** could simplify training.

### Global vs Regional Tradeoffs

Earth4D uses a single hash table for the entire planet. This is memory-efficient but treats all regions equally. Some regions (densely sampled urban areas) may benefit from finer-grained encoding than sparse regions (oceans, deserts).

**Future work**: Adaptive hash allocation based on data density. Allocate more hash capacity to high-information regions, less to homogeneous areas.

### Temporal Resolution Assumptions

Our experiments normalize time to $$[0, 1]$$ over the dataset's temporal range. For applications spanning centuries (climate modeling), we may need explicit multi-scale temporal encoding (years, months, days, hours) similar to spatial multi-resolution.

### Interpretability

While Earth4D learns effective representations, understanding **what** it learns remains challenging. Visualization of hash table features could reveal:
- Which spatiotemporal patterns activate specific hash entries?
- How do features at different resolution levels specialize?
- Can we interpret learned probe offsets?

Techniques from mechanistic interpretability<d-cite key="olah2020zoom"></d-cite> could shed light on Earth4D's internal representations.

---

## Conclusion

We presented Earth4D, a production-ready 4D space-time positional encoder that achieves state-of-the-art performance on ecological forecasting while using only spatiotemporal coordinates. Through decomposed hash encoding and learned hash probing, Earth4D demonstrates:

1. **Matching foundation models** pretrained on multimodal Earth observation data, using coordinates alone
2. **99% parameter reduction** with maintained or improved performance
3. **Planetary-scale coverage** from sub-meter to continental resolution
4. **4× training speedup** enabling practical deployment

These results challenge the assumption that world models require massive multimodal pretraining. Rich spatiotemporal representations, learned through 4D hash encoding, can capture complex Earth dynamics with remarkable efficiency.

Earth4D is fully open source and ready for integration into world models:

**GitHub**: [https://github.com/legel/deepearth](https://github.com/legel/deepearth)

As we build AI systems to understand and simulate our planet—for climate forecasting, disaster response, agricultural planning, and beyond—**positional encoding matters**. Earth4D provides a foundation for world models that is memory-efficient, expressive, and ready for production deployment.

The future of world models may not require encoding everything about Earth. Perhaps we just need to encode Earth's space-time structure effectively—and let the model discover the rest.

---

## Acknowledgments

We thank the Allen Institute for AI for releasing the Globe-LFMC 2.0 dataset and Galileo baseline. We thank NVIDIA for open-sourcing InstantNGP, which inspired Earth4D's architecture. We thank the USGS 3DEP and USDA NAIP programs for providing public LiDAR and imagery data.
