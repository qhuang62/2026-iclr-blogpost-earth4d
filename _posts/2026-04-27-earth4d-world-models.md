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
  - name: Building Blocks
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

Imagine predicting wildfire risk across California's 163,000 square miles using nothing but coordinates: no satellite imagery, no weather data, no topography. Just latitude, longitude, elevation, and time. Impossible?

That's what we thought too—until Earth4D proved otherwise.

World models—AI systems that learn to simulate and predict complex spatiotemporal dynamics—have emerged as a powerful paradigm for understanding our planet. From climate forecasting to disaster response, these models traditionally rely on massive multimodal datasets: satellite imagery, weather reanalysis, topographic maps, and sensor networks. The assumption has been that **rich inputs are necessary for rich predictions**.

But at the heart of all Earth observation data lies a more fundamental structure: **space and time**. Every measurement, every pixel, every sensor reading exists at specific coordinates (x,y,z,t). What if we could capture the essence of Earth's dynamics by encoding this 4D spatiotemporal structure more effectively?

This is the central question behind **Earth4D**, a production-ready 4D space-time positional encoder developed as part of the DeepEarth world model. Earth4D extends multi-resolution hash encoding from 3D graphics to four dimensions, achieving surprising results:

- **Surpasses multimodal foundation models** using only (x,y,z,t) coordinates—no satellite imagery, weather data, or topography required
- **99% parameter reduction** (724M → 5M) with 4× training speedup while maintaining strong performance
- **Planetary-scale coverage** from sub-meter to continental scale, with temporal precision from sub-second to centuries

More importantly, Earth4D challenges a fundamental assumption in world modeling: rather than requiring massive multimodal pretraining, we can achieve state-of-the-art performance by learning rich spatiotemporal representations from coordinates alone. The key is getting the positional encoding right.

### Earth4D in the DeepEarth World Model

Earth4D is a core component of **DeepEarth**, a self-supervised multi-modal world model for planetary-scale Earth observation. While DeepEarth can process diverse data types—satellite imagery, sensor readings, text descriptions—its secret weapon is Earth4D's rich spatiotemporal encoding.

<figure>
  <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/deepearth_main_figure.png' | relative_url }}" alt="DeepEarth Architecture Overview" style="width: 100%;">
  <figcaption><b>DeepEarth World Model Architecture.</b> Multi-modal data (images, text, sensor data) sampled around spatiotemporal events are encoded by modality-specific encoders and fused with Earth4D space-time embeddings. These universal tokens are jointly processed through an autoencoder that learns to reconstruct and simulate masked data. Earth4D provides the spatiotemporal grounding that enables the model to reason about where and when phenomena occur. (Figure from DeepEarth paper)</figcaption>
</figure>

In this blog post, we focus specifically on **Earth4D**—the 4D positional encoder—and demonstrate that even in isolation, without multimodal data, it achieves remarkable performance. This validates that Earth4D captures genuinely useful spatiotemporal structure, not just auxiliary context for other modalities.

---

## Building Blocks: What We Built Upon

Earth4D combines several existing techniques in a novel way for planetary-scale Earth observation. Before diving into our work, it's important to acknowledge the foundational methods we adapted:

**InstantNGP (Müller et al., 2022)**<d-cite key="muller2022instant"></d-cite>: Introduced multi-resolution hash encoding for 3D neural graphics. Their insight: compress spatial features into fixed-size hash tables across multiple resolution levels. We extend this from 3D to 4D.

**Grid4D (Xu et al., 2024)**<d-cite key="xu2024grid4d"></d-cite>: Pioneered decomposing 4D space-time into four 3D grids (xyz, xyt, yzt, xzt) for dynamic Gaussian splatting. We adopt their decomposition strategy wholesale, adapting it from computer graphics to Earth observation.

**Learned Hash Probing (Takikawa et al., 2023)**<d-cite key="takikawa2023compact"></d-cite>: Developed at NVIDIA Toronto AI Lab to intelligently resolve hash collisions through learned probe offsets. We apply their technique to enable extreme compression for edge deployment.

**Our contribution**: We didn't invent these components. Instead, Earth4D demonstrates that combining these graphics techniques—originally designed for rendering virtual scenes—works remarkably well for modeling our actual planet. The novelty is in the application domain, scale (planetary), and empirical validation (state-of-the-art ecological forecasting with coordinates alone).

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
  <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/earth4d_spacetime_encoder.png' | relative_url }}" alt="Earth4D Space-Time Positional Encoding" style="width: 100%;">
  <figcaption><b>Earth4D Space-Time Positional Encoding.</b> A planetary-scale 4D encoder with fully decomposable spatio-temporal representation. Geographic coordinates (latitude, longitude, elevation) are converted to Earth-Centered Earth-Fixed (ECEF) coordinates and normalized. Four grids (xyz, xyt, yzt, xzt) are each learned in 3D space and computed in parallel. Each grid has multiple resolution levels, enabling deep learning of complex joint distributions in multi-modal data across space-time scales. (Figure from DeepEarth paper)</figcaption>
</figure>

**Credit where credit is due**: This decomposition strategy comes directly from **Grid4D**<d-cite key="xu2024grid4d"></d-cite> (Xu et al., 2024), developed for high-fidelity dynamic Gaussian splatting. Grid4D pioneered the idea of decomposing 4D space-time into four 3D grids (xyz, xyt, yzt, xzt) rather than using a single 4D grid. We adapted their decomposition approach from computer graphics to planetary-scale Earth observation, but the core architectural insight is theirs.

By using three orthogonal spatiotemporal projections, we capture how spatial patterns evolve over time from different perspectives. The XYZ grid provides pure spatial context, while XYT, YZT, and XZT encode temporal dynamics in complementary subspaces—exactly as Grid4D designed.

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

**Intuition first**: Think of Earth4D like a hierarchical address system. At the coarsest level, we divide the planet into large regions (like countries). At finer levels, we divide into cities, neighborhoods, streets, and buildings. Each location gets multiple "addresses" at different zoom levels.

The challenge: storing a unique feature for every possible location at every resolution level would require astronomical memory. The solution: **hash encoding** compresses this into a fixed-size table using a clever mathematical trick.

**The process** at each resolution level $$L$$:

1. **Discretize**: Convert continuous coordinates to grid positions
   `pos_grid = floor(coordinate × resolution_L)`

2. **Index or Hash**:
   - Coarse levels (few grid cells): Store directly, no collisions
   - Fine levels (many grid cells): Hash multiple positions to the same memory slot

   ```
   if grid_size ≤ hashmap_size:
       index = pos_grid.x + pos_grid.y × stride_y + pos_grid.z × stride_z
   else:
       index = hash(pos_grid) mod hashmap_size
   ```

3. **Interpolate**: Blend features from 8 surrounding grid corners (trilinear interpolation)

The hash function mixes coordinates using XOR with large primes:

$$\text{hash}(\mathbf{p}) = \bigoplus_{d=1}^{D} p_d \cdot \pi_d \pmod{T}$$

**Why does this work?** The hash function scrambles nearby positions to distant memory locations, preventing clusters of similar coordinates from competing for the same slot. Think of it like distributing people across hotel rooms—random assignment prevents overcrowding.

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
  <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/hash_collision_1m_simulation.png' | relative_url }}" alt="Earth4D Hash Collision Analysis" style="width: 100%;">
  <figcaption><b>Earth4D Hash Collision Analysis.</b> Simulation results for 1M points across 10 spatiotemporal distribution scenarios (rows) and 24 resolution levels (columns) for all four grids (XYZ, XYT, YZT, XZT). Yellow indicates high collision rates, purple indicates low collision rates. Fine levels (right side, high resolution) show expected 2-4% collision rates across most scenarios. Extreme spatial/temporal clustering scenarios show higher collisions at intermediate levels where grid resolution exceeds hash table capacity. (Figure adapted from DeepEarth paper Figure 6)</figcaption>
</figure>

**Are collisions bad?** Not necessarily. The hash encoding literature<d-cite key="muller2022instant"></d-cite> shows that downstream networks (MLPs) can learn to disambiguate collisions when they're relatively rare. The hash function's randomness actually acts as a form of regularization.

However, **catastrophic collision patterns** destroy information. If all temporal variations at a location map to the same index, we lose the ability to model temporal dynamics.

### The uint32 Overflow Discovery

**A debugging story**: During development, our temporal predictions were inexplicably bad. The model couldn't distinguish between summer and winter at the same location—as if time didn't exist.

Investigation revealed catastrophic collision patterns:
- **Level 8**: 100% collision rate (41,261 coordinates mapped to only ~978 unique indices)
- **Levels 13-19**: 99.9% collision rate (coordinates with different timestamps but identical spatial positions mapped to the same memory slot)

This violated fundamental expectations—collision rates should decrease as we move to finer resolutions, not stay constant at 99.9%. The hash function was broken, but how?

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

> **Key Takeaway**: When implementing multi-resolution encodings, always use 64-bit arithmetic for index calculations, even if final indices fit in 32 bits. Intermediate products can overflow silently, destroying temporal/spatial information in ways that appear as poor model performance rather than obvious crashes.

### Collision Patterns Across Scales

Even with the overflow bug fixed, hash collisions are inherent to memory-efficient encoding. We profiled collision rates across 10 different spatiotemporal data distributions:

1. **Uniform Random**: Global Earth surface sampling
2. **Continental Sparse**: Sparse coverage of North America
3. **City-Scale Cluster**: 10km × 10km dense sampling
4. **Building-Scale**: Single 10m × 10m area over time
5. **Time Series**: Fixed locations sampled repeatedly over time

Results showed collision rates ranging from 0% (coarse levels) to 2-4% (fine levels), with expected power-of-2 artifacts at level 23 (67M grid cells, 4M hash entries = exact 16× ratio).

While 2-4% collision rate is acceptable for many applications, we wanted to push further: **Can we reduce hash table size even more while maintaining quality?**

The answer lies in making collisions smarter, not just rarer.

---

## Learned Hash Probing

Standard hash encoding treats all collisions equally—coordinates compete randomly for memory slots. But what if we could teach the model to **intelligently allocate** memory based on the actual data distribution?

This is where learned hash probing becomes crucial for achieving extreme compression.

### How Learned Probing Works

**The collision problem visualized**: Imagine multiple coordinates competing for the same memory slot. Standard hash encoding assigns them randomly—some get lucky and land in empty slots, others collide and degrade quality.

**Learned probing solution**: Give the model multiple candidate slots and let it learn which to use. It's like having backup hotel rooms—if your first choice is crowded, the system learns to route you to a better alternative.

**Credit**: Learned hash probing was developed by **Takikawa et al. (2023)** at NVIDIA Toronto AI Lab<d-cite key="takikawa2023compact"></d-cite> as an improvement to InstantNGP. We did not invent this technique—we applied it to Earth observation. Their method uses **dual hashing with learned offsets**:

$$\text{index} = N_p \times h_1(\mathbf{x}) + \mathcal{D}_c[h_2(\mathbf{x})]$$

Breaking this down:
- $$h_1(\mathbf{x})$$: Primary hash (rough neighborhood)
- $$h_2(\mathbf{x})$$: Secondary hash selecting among $$N_p$$ candidates (typically 4-32 options)
- $$\mathcal{D}_c$$: **Learned codebook**—the model discovers which offsets work best for the data

**The learning process**: Initially, the model randomly distributes data across all candidate slots. During training, gradients flow backward through a **straight-through estimator**—the forward pass selects a discrete index (hard choice), but the backward pass distributes gradients across all candidates weighted by softmax probabilities (soft gradients).

Over thousands of training steps, the model learns patterns like:
- "Densely sampled urban regions should use probes 0-7"
- "Sparse oceanic regions can share probe 31"
- "Temporal clusters need dedicated probes to avoid interference"

This **data-adaptive collision resolution** outperforms fixed hash functions because it learns the actual distribution of your training data rather than assuming uniform randomness.

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

**Why does compression improve performance?** We hypothesize that extreme compression acts as regularization, similar to how restricting a student's note-taking forces deeper understanding rather than verbatim transcription.

**Concrete analogy**: Imagine learning geography with different-sized notebooks:
- **Large notebook (724M params)**: Write down every detail about every location → risk memorizing specific training examples
- **Small notebook (5M params)**: Must capture essential patterns → forced to learn "coastal regions are moist in winter" rather than "GPS coordinate 37.7749°N has LFMC 87% on Jan 15, 2019"

The forced sharing of hash table entries encourages the model to discover **reusable spatiotemporal features** rather than overfitting to training coordinates. This parallels how dropout or weight decay improve generalization—constraints prevent memorization.

---

## Experimental Validation

We evaluate Earth4D through three research questions, each testing a critical capability for world models:

> **Q1: Can coordinates alone match multimodal foundation models?**
> Tests whether spatiotemporal encoding captures enough information to compete with satellite imagery and weather data
> → **Result**: Earth4D surpasses Galileo foundation model (12.4 vs 12.6 MAE on ecological forecasting)
>
> **Q2: Can we achieve 99% parameter reduction while maintaining performance?**
> Tests whether learned hash probing enables extreme compression
> → **Result**: 724M → 5M parameters with improved R² (0.582 → 0.668)
>
> **Q3: Can Earth4D learn arbitrary spatiotemporal functions?**
> Tests generality by predicting RGB pixels from elevation alone
> → **Result**: 18% lower loss with learned probing on Houston wetlands reconstruction

Let's examine each in detail.

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
  <div style="display: flex; flex-direction: column; gap: 10px;">
    <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/geospatial_error_map_epoch_2500.png' | relative_url }}" alt="LFMC Geographic Error Distribution" style="width: 100%;">
    <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/temporal_predictions_epoch_2500.png' | relative_url }}" alt="LFMC Temporal Predictions" style="width: 100%;">
  </div>
  <figcaption><b>Earth4D LFMC Prediction Performance.</b> <b>Top</b>: Geographic error distribution across CONUS shows low error in well-sampled regions, with median absolute error of 7.1 percentage points. <b>Bottom</b>: Temporal predictions (black line) closely track ground truth LFMC measurements (gray distribution) across seasons (2017-2023), demonstrating Earth4D's ability to capture seasonal vegetation moisture dynamics using only spatiotemporal coordinates. (Figures from DeepEarth paper)</figcaption>
</figure>

**How is this possible?** Earth4D learns that spatiotemporal coordinates encode surprisingly rich information about vegetation moisture:

**Spatial patterns**:
- Coastal California (37°N, -122°W, low elevation) → high LFMC in winter (Pacific moisture)
- Arizona desert (33°N, -111°W, moderate elevation) → low LFMC year-round (arid climate)
- Pacific Northwest (47°N, -122°W, high elevation) → consistently high LFMC (temperate rainforest)

**Temporal patterns**:
- Summer months (June-August) → lower LFMC across most regions (heat stress, reduced precipitation)
- Winter months (December-February) → higher LFMC in Mediterranean climates (wet season)
- Spring months (March-May) → peak LFMC in temperate zones (snowmelt, spring rains)

**Elevation effects**:
- Low elevation (<500m) → follows regional climate patterns directly
- Mid elevation (500-1500m) → extended moisture retention from orographic precipitation
- High elevation (>1500m) → snowpack buffering creates delayed moisture dynamics

**Spatiotemporal interactions**:
The same location exhibits different LFMC at different times, and the same time period exhibits different LFMC at different locations. Earth4D's multi-resolution structure captures both:
- **Coarse levels (398km/cell)**: Encode climate zones—Mediterranean, desert, temperate, tropical
- **Fine levels (4.75cm/cell)**: Encode microclimate variations—north-facing vs south-facing slopes, proximity to water sources, local topographic moisture traps

Crucially, the **species embedding provides botanical context**. Earth4D learns that chaparral shrubs in coastal California have different moisture dynamics than pine trees in the same location, despite identical coordinates. The model discovers species-specific water use strategies encoded in the learnable embedding.

This result challenges a fundamental assumption: **you don't always need satellite imagery and weather data if your positional encoding is expressive enough**. Rich spatiotemporal structure, learned through multi-resolution hash encoding, captures climate patterns that manifest in vegetation moisture.

**Important caveat**: This doesn't mean coordinates are universally superior to multimodal data. Earth4D succeeds on LFMC prediction because:
1. LFMC correlates strongly with climate zones, which are fundamentally spatiotemporal
2. The dataset spans multiple years, allowing temporal patterns to be learned
3. Species identity provides crucial botanical context

For tasks requiring fine-grained visual understanding (crop disease detection, infrastructure damage assessment), satellite imagery would likely remain essential. Earth4D's success highlights that **some Earth observation tasks may be overengineered**, relying on expensive multimodal data when simpler coordinate-based approaches suffice.

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
  <img src="{{ 'assets/img/2026-04-27-earth4d-world-models/rgb_reconstruction_houston.png' | relative_url }}" alt="RGB Reconstruction from LiDAR Elevation" style="width: 100%;">
  <figcaption><b>RGB Reconstruction from LiDAR Elevation.</b> Houston coastal wetlands, 2018. <b>Left to right</b>: LiDAR height (input), ground truth RGB, baseline reconstruction (no probing), learned probing reconstruction (18% lower loss). Earth4D learns to infer RGB pixel values from elevation and coordinates alone, capturing correlations between topography and land cover. Water bodies (low, flat elevation) appear blue/green, vegetation (moderate elevation) appears green, and urban areas (varied elevation) show gray/brown tones. (Data from USGS 3DEP and USDA NAIP)</figcaption>
</figure>

The model learns complex correlations:
- **Water bodies** (low elevation, flat) → blue/green hues
- **Vegetation** (moderate elevation, rough terrain) → green
- **Urban areas** (high elevation variance) → gray/brown
- **Coastal transitions** (elevation gradients) → color gradients

Learned probing significantly improves reconstruction quality, especially in fine-detail regions like coastline boundaries and vegetation patches.

**Limitations of this experiment**: While visually impressive, the RGB reconstruction task has significant limitations:
1. **Overfitting to local correlations**: The model learns Houston-specific patterns (local vegetation types, soil colors, urban development). It wouldn't generalize to other regions with different elevation-color relationships (e.g., desert regions where low elevation doesn't imply water).
2. **Limited semantic understanding**: The model doesn't understand "what" it's reconstructing—it's purely statistical correlation between elevation and color in this specific dataset.
3. **Validation loss vs perceptual quality**: 18% loss reduction doesn't necessarily mean 18% better visual quality. Some fine details may improve while overall structure remains similar.

This experiment demonstrates Earth4D's ability to fit **implicit spatiotemporal functions**, but doesn't prove it learns generalizable representations beyond the training distribution. For production applications, task-specific validation would be essential.

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

The 99% parameter reduction (724M → 5M) makes Earth4D viable for real-world deployment scenarios that were previously impossible:

**Satellite onboard processing**: Modern satellites like Planet Labs' Doves have limited computational resources. Earth4D's 5M parameter compressed model (850MB memory) can run on satellite GPUs, enabling real-time wildfire risk assessment as imagery is captured—eliminating the latency of downlinking data to ground stations.

**Mobile disaster response**: During wildfires or floods, field teams often operate in areas with limited connectivity. A tablet or smartphone running Earth4D can provide location-specific risk predictions (fire spread likelihood, flood extent forecasts) using only GPS coordinates and local time—no network connection required.

**IoT sensor networks**: Environmental monitoring stations deployed across forests or agricultural lands often run on solar power with limited energy budgets. Earth4D's 4× faster inference enables battery-powered edge devices to perform hourly moisture monitoring, triggering alerts when fire risk exceeds thresholds.

**Developing regions**: Many countries lack access to expensive satellite imagery subscriptions or high-performance computing clusters. Earth4D democratizes Earth observation AI by requiring only coordinate data—freely available from GPS—rather than costly multimodal datasets.

This shifts world models from datacenter-scale infrastructure to **ubiquitous deployment**, enabling real-time decision-making where it matters most: on satellites, in the field, and in resource-constrained environments.

### 3. Learned Probing as Universal Compression

**Broader applications**: Takikawa et al.'s learned hash probing technique isn't specific to Earth observation—it's a **general method for compressing hash-based neural representations**. Beyond our Earth4D application, it has been used for:
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

When we set out to build Earth4D, we had a simple hypothesis: **spatiotemporal structure matters more than we think**. The conventional wisdom in Earth observation AI emphasizes collecting more modalities—higher resolution imagery, more spectral bands, denser weather data, richer metadata. But what if the key isn't adding more data types, but rather encoding the fundamental structure—space and time—more effectively?

Earth4D validates this hypothesis. Through decomposed 4D hash encoding and learned hash probing, it achieves state-of-the-art ecological forecasting using only (x,y,z,t) coordinates. No satellite imagery. No weather reanalysis. No topography. Just position in space-time.

**Key results**:
1. **Surpasses multimodal foundation models** pretrained on diverse Earth observation data (12.4 vs 12.6 MAE on Globe-LFMC 2.0)
2. **99% parameter reduction** (724M → 5M) with maintained or improved performance
3. **4× training speedup** enabling edge deployment on satellites, mobile devices, and IoT sensors
4. **Planetary-scale coverage** from sub-meter (4.75cm) to continental (398km) resolution, across sub-second to century timescales

These results have implications beyond Earth observation. They suggest that for spatiotemporal problems more broadly—video understanding, robotics, autonomous navigation—**positional encoding deserves first-class treatment**, not just auxiliary context for pixel features.

**The path forward**: Earth4D is a component of the DeepEarth world model, where it provides spatiotemporal grounding for multimodal encoders. We envision future world models that:
- **Bootstrap** from coordinates when multimodal data is unavailable (sparse regions, historical periods, privacy-sensitive applications)
- **Compress** through learned hash probing, enabling deployment beyond datacenters
- **Generalize** by learning spatiotemporal patterns that transfer across tasks

As we build AI systems to understand and simulate our planet—for climate adaptation, disaster response, agricultural resilience, and ecosystem monitoring—**how we encode space and time fundamentally shapes what patterns models can discover**.

Earth4D demonstrates that with the right positional encoding, coordinates alone can capture the essence of Earth's spatiotemporal dynamics. The future of world models may not require encoding everything about our planet. Perhaps we just need to encode the structure of space-time effectively—and let the model discover the rest.

---

## Acknowledgments

We thank the Allen Institute for AI for releasing the Globe-LFMC 2.0 dataset and Galileo baseline. We thank NVIDIA for open-sourcing InstantNGP, which inspired Earth4D's architecture. We thank the USGS 3DEP and USDA NAIP programs for providing public LiDAR and imagery data.
