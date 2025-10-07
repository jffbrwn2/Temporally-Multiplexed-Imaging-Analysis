# Fluorescence Unmixing for Non-Spectral Multiplexing

Python implementation of fluorescence unmixing for reversibly switchable fluorescent proteins (rsFPs) with distinct decay kinetics.

## Overview

This tool performs linear unmixing of fluorescence time-series data to separate contributions from different fluorescent proteins based on their unique decay kinetics rather than spectral properties.

### Supported Fluorophores

- **AP1-EGFP**: Constant/baseline (non-photoswitchable)
- **DUSP5-Skylan**: Medium decay rate
- **EGR1-Dronpa**: Fast decay
- **FOS-GreenFast**: Very fast decay
- **SARE-rsFastLime**: Medium-fast decay

## Installation

### Requirements

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) package installer (recommended) or pip

### Using uv (Recommended)

1. Install uv if you don't have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd fluorescence_unmixing_public
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

### Using pip

```bash
pip install -e .
```

## Usage

### Basic Command

```bash
python unmixing_python.py
```

### Command-Line Options

```bash
# Use fitted exponential reference curves instead of population averages
python unmixing_python.py --fitted

# Constrain EGFP background channel to 0-500 range
python unmixing_python.py --constrain-background

# Combine both options
python unmixing_python.py --fitted --constrain-background
```

### Configuration

Edit the `main()` function in `unmixing_python.py` to set your paths:

```python
base_path = "/path/to/your/data"
ref_path = os.path.join(base_path, "datRef.csv")
green_folder = os.path.join(base_path, "green_mat_curves")
red_mask_folder = os.path.join(base_path, "red_masks")
output_dir = os.path.join(base_path, "unmixing_output")
```

### Input Data Format

The script expects:

1. **Reference curves** (`datRef.csv` or `datRefK.csv`):
   - Tab-separated values
   - Each column represents a fluorophore's decay curve
   - Normalized to peak intensity

2. **Green curve files** (`*green_curve_2.mat`):
   - MATLAB `.mat` files containing `datPic` variable
   - 3D array: (timeframes, height, width)
   - Format: `{WellID}_ROI_{ROI#}_green_curve_2.mat`

3. **Red mask files** (`*red_mask.mat`):
   - MATLAB `.mat` files containing `red_image` variable
   - 2D binary mask indicating active pixels
   - Format: `{WellID}_ROI_{ROI#}_red_mask.mat`

### Output Files

For each ROI, the script generates:

1. **Comprehensive analysis plot** (`{WellID}-ROI{ROI#}-comprehensive.png`):
   - Bar plot of unmixed coefficients
   - Fitted vs. measured curves
   - Individual component contributions

2. **Bar plot** (`{WellID}-ROI{ROI#}-barplot.png`):
   - Simple bar chart of unmixing results

3. **Multi-channel TIFF** (`{WellID}-ROI{ROI#}-curve2-multichannel.tif`):
   - 5-channel image with separated fluorophore signals

4. **NumPy compressed file** (`{WellID}-ROI{ROI#}-c2-outMAT.npz`):
   - Raw unmixing coefficients for further analysis

## Algorithm Details

### Non-Negative Least Squares (NNLS)

The unmixing uses non-negative least squares optimization:

```
minimize ||A·x - b||²
subject to x ≥ 0
```

Where:
- `A` = reference decay curves matrix
- `x` = unknown mixing coefficients
- `b` = measured fluorescence time series

### Optional Background Constraint

With `--constrain-background`, the EGFP channel is bounded to [0, 500] to better estimate background:

```
minimize ||A·x - b||²
subject to 0 ≤ x₀ ≤ 500 and xᵢ ≥ 0 for i > 0
```

## Methodology

This implements **non-spectral multiplexing** using reversibly switchable fluorescent proteins. Each protein has unique decay kinetics when imaged over time:

**Fluorescence decay model:**
```
f(t) = (1-C)·e^(-k·t) + C
```

Where:
- `k` = decay rate constant (unique per fluorophore)
- `C` = plateau/baseline fluorescence
- `t` = time (frame number)

## Citation

If you use this tool in your research, please cite:

[Paper citation to be added]

## License

[License to be specified]

## Contact

For questions or issues, please open an issue on the GitHub repository.

## Troubleshooting

### Common Issues

1. **File not found errors**: Check that your paths are correct and files follow the naming convention
2. **No matching files**: Ensure green curve and red mask files have matching well IDs and ROI numbers
3. **Memory errors**: Process fewer files at once or reduce image dimensions
4. **NFS timeout warnings**: The script has built-in retry logic for network filesystem issues

### Getting Help

- Check the output messages for specific error details
- Verify your input data format matches the expected structure
- Ensure all dependencies are correctly installed
