# Ultimate AI/ML Trading Workstation - PC Configuration Guide

## Overview

This document specifies the **best-in-class PC configuration** for an extreme AI/ML trading workstation designed to handle:

- **AI/ML Model Training** on 10 years of tick data (billions of rows)
- **Deep Learning Model Training** on 10 years of tick data
- **LLM Training/Fine-Tuning** on 10 years of tick data
- **OpenClaw** for high-volume trading activity
- **Stock Scanner** scanning thousands of stocks in real-time
- **Auto-Execution Systems** for automated trade execution
- **NinjaTrader** for charting, analysis, and order management
- **Simultaneous operation** of all the above workloads

---

## Workload Analysis

| Workload | CPU Demand | GPU Demand | RAM Demand | Storage I/O | Network |
|---|---|---|---|---|---|
| ML Models (10yr tick data) | Very High | High | Very High | Very High | Low |
| Deep Learning (10yr tick data) | High | **Extreme** | Very High | Very High | Low |
| LLM Training (10yr tick data) | High | **Extreme** | **Extreme** | Very High | Low |
| OpenClaw Trading | High | Low | High | High | **Critical** |
| Stock Scanner (1000s stocks) | Very High | Medium | Very High | High | **Critical** |
| Auto-Execution | High | Low | High | High | **Critical** |
| NinjaTrader | Medium | Low | Medium | Medium | High |

---

## TIER 1: ULTIMATE BUILD (No Budget Limit)

**Estimated Cost: $35,000 - $55,000+**

This is the absolute best configuration for maximum performance across all workloads.

### CPU (Processor)

| Component | Specification |
|---|---|
| **CPU** | **AMD Threadripper PRO 7995WX** |
| Cores/Threads | 96 Cores / 192 Threads |
| Base Clock | 2.5 GHz |
| Boost Clock | 5.1 GHz |
| L3 Cache | 384 MB |
| TDP | 350W |
| PCIe Lanes | 128 PCIe 5.0 Lanes |

**Why this CPU:**
- 96 cores handle massive parallel data preprocessing of 10 years of tick data
- 128 PCIe 5.0 lanes support multiple GPUs at full bandwidth (critical for multi-GPU LLM training)
- 8-channel DDR5 memory supports up to 2TB RAM for loading entire datasets into memory
- Exceptional single-thread boost (5.1 GHz) for NinjaTrader and trading platforms that are single-thread sensitive

### Motherboard

| Component | Specification |
|---|---|
| **Motherboard** | **ASUS Pro WS WRX90E-SAGE SE** (or equivalent WRX90 board) |
| Chipset | AMD WRX90 |
| Memory Slots | 8x DDR5 DIMM (R-DIMM/ECC supported) |
| Max Memory | 2 TB DDR5 |
| PCIe Slots | 7x PCIe 5.0 x16 |
| M.2 Slots | 4x M.2 PCIe 5.0 |
| Networking | Dual 10GbE LAN |
| Form Factor | E-ATX / SSI-EEB |

**Why this board:**
- Full 8-channel DDR5 memory support
- 7 PCIe 5.0 x16 slots for multi-GPU configurations
- Dual 10GbE for ultra-low-latency market data feeds
- Enterprise-grade VRM for sustained heavy loads

### GPU (Graphics Processing Units)

| Component | Specification |
|---|---|
| **Primary GPUs** | **2x NVIDIA RTX 6000 Ada Generation** (or 2x NVIDIA H100 PCIe if budget allows) |
| VRAM per GPU | 48 GB GDDR6X (RTX 6000 Ada) / 80 GB HBM3 (H100) |
| Total VRAM | 96 GB (RTX 6000 Ada) / 160 GB (H100) |
| CUDA Cores | 18,176 per GPU (RTX 6000 Ada) |
| Tensor Cores | 568 per GPU (4th Gen, RTX 6000 Ada) |
| NVLink | Supported (RTX 6000 Ada) |
| FP16 Performance | 91.1 TFLOPS per GPU (RTX 6000 Ada) |

**Why these GPUs:**
- **For LLM Training:** 48GB+ VRAM per GPU is essential; LLM fine-tuning on large tick-data corpora requires massive VRAM
- **For Deep Learning:** Tensor Cores accelerate transformer, LSTM, and CNN training by 10-20x
- **NVLink bridge** allows GPUs to share memory for models that exceed single-GPU VRAM
- **Multi-GPU scaling:** 2 GPUs provide near-linear speedup for distributed training (PyTorch DDP, DeepSpeed, etc.)

#### Alternative GPU Option (Best Price/Performance)

| Component | Specification |
|---|---|
| **Budget GPUs** | **2x NVIDIA RTX 4090** |
| VRAM per GPU | 24 GB GDDR6X |
| Total VRAM | 48 GB |
| CUDA Cores | 16,384 per GPU |
| Tensor Cores | 512 per GPU (4th Gen) |
| FP16 Performance | 82.6 TFLOPS per GPU |

> **Note:** RTX 4090 is excellent for deep learning but only 24GB VRAM limits LLM training. Use gradient checkpointing, LoRA, or QLoRA techniques to work within VRAM limits.

### RAM (System Memory)

| Component | Specification |
|---|---|
| **RAM** | **512 GB DDR5-5600 ECC R-DIMM** (8x 64GB) |
| Type | DDR5 Registered ECC |
| Speed | 5600 MT/s |
| Channels | 8-channel |
| Expandable To | 2 TB (swap to 8x 256GB modules) |

**Why 512GB:**
- 10 years of tick data for a single instrument at 1-tick resolution can be **50-200+ GB** in memory
- Scanning thousands of stocks simultaneously requires loading multiple datasets
- ML feature engineering (rolling windows, indicators, etc.) can 3-5x the data size in memory
- LLM tokenized datasets plus model weights plus optimizer states consume enormous RAM
- NinjaTrader + OpenClaw + Scanner all running simultaneously need dedicated memory pools

**Memory Budget Breakdown:**
| Application | Estimated RAM Usage |
|---|---|
| 10yr tick data (loaded) | 50-200 GB |
| ML/DL model training overhead | 32-64 GB |
| LLM training (CPU offload) | 64-128 GB |
| OpenClaw | 8-16 GB |
| Stock Scanner (1000s stocks) | 32-64 GB |
| NinjaTrader | 4-8 GB |
| Auto-Execution Engine | 4-8 GB |
| OS + System Services | 8-16 GB |
| **Total Estimated** | **~200-500 GB** |

### Storage

| Drive | Specification | Purpose |
|---|---|---|
| **OS/Boot Drive** | **Samsung 990 Pro 2TB PCIe 4.0 NVMe** | Windows OS, NinjaTrader, trading platforms |
| **AI/ML Working Drive** | **2x Samsung PM1743 3.84TB PCIe 5.0 NVMe (RAID 0)** | Active model training, datasets, checkpoints |
| **Tick Data Storage** | **Solidigm D7-P5810 6.4TB PCIe 4.0 NVMe** | 10 years tick data archive, high-endurance |
| **Backup/Cold Storage** | **2x Seagate Exos X24 24TB HDD (RAID 1 Mirror)** | Data backup, model archives |

**Storage Performance Targets:**

| Drive | Sequential Read | Sequential Write | Random 4K IOPS |
|---|---|---|---|
| OS Drive | 7,450 MB/s | 6,900 MB/s | 1,400K |
| AI/ML Working (RAID 0) | 14,000+ MB/s | 12,000+ MB/s | 2,500K+ |
| Tick Data | 7,000 MB/s | 6,500 MB/s | 1,200K |

**Why this storage layout:**
- **PCIe 5.0 NVMe RAID 0** for AI/ML training eliminates storage as a bottleneck during data loading
- **High-endurance enterprise NVMe** for tick data handles constant read/write cycles
- **Separate drives** prevent I/O contention between trading platforms and model training
- **HDD RAID 1 mirror** protects irreplaceable historical data

### Power Supply

| Component | Specification |
|---|---|
| **PSU** | **Corsair AX1600i** or **be quiet! Dark Power Pro 13 1600W** |
| Wattage | 1600W |
| Efficiency | 80+ Titanium |
| Modularity | Fully Modular |
| Rails | Single 12V Rail |

**Power Budget:**
| Component | Power Draw |
|---|---|
| CPU (TR PRO 7995WX) | 350W (peak 500W) |
| 2x RTX 6000 Ada | 600W (2x 300W) |
| 512GB DDR5 RAM | ~50W |
| Storage (all drives) | ~40W |
| Fans/Cooling | ~30W |
| **Total Estimated** | **~1,070W peak** |
| **PSU Headroom (50%)** | **1,600W** |

### CPU Cooling

| Component | Specification |
|---|---|
| **Cooler** | **Noctua NH-U14S TR5-SP6** or **Arctic Liquid Freezer III 420 (with sTR5 bracket)** |
| Type | Air (Noctua) or 420mm AIO Liquid (Arctic) |

**Recommendation:** For sustained 96-core loads during model training, a **420mm AIO liquid cooler** is preferred. The Threadripper PRO 7995WX generates significant heat under all-core loads.

### Case

| Component | Specification |
|---|---|
| **Case** | **Fractal Design Torrent XL** or **Corsair 7000D Airflow** |
| Form Factor | Full Tower / Super Tower |
| Fan Support | 420mm top + 360mm front radiator support |
| Drive Bays | Multiple 3.5" + 2.5" bays |

**Why these cases:**
- Massive airflow for high TDP components
- Room for E-ATX motherboard + full-length dual GPUs
- Multiple radiator mounting positions
- Excellent cable management for clean builds

### Networking

| Component | Specification |
|---|---|
| **Primary NIC** | Onboard Dual 10GbE (from motherboard) |
| **Secondary NIC** | **Mellanox ConnectX-6 25GbE SFP28** (optional) |
| **Router/Switch** | **Ubiquiti UniFi Dream Machine Pro** + **10GbE switch** |

**Why advanced networking:**
- Real-time market data feeds for thousands of stocks require low-latency, high-bandwidth networking
- Auto-execution systems need minimal network jitter
- Separate NICs allow traffic isolation (trading vs. general internet)

### Monitors

| Component | Specification |
|---|---|
| **Primary Monitor** | **Samsung Odyssey Ark 55" 4K 165Hz** or **LG 45GS96QB 45" OLED 240Hz** |
| **Secondary Monitors** | **3x Dell U2723QE 27" 4K IPS** |
| **Total Screens** | 4 monitors |

**Why this setup:**
- Large primary for NinjaTrader charts and analysis
- Secondary monitors for: Stock Scanner, OpenClaw, Auto-Execution dashboard
- 4K resolution provides pixel density for dense financial data

### UPS (Uninterruptible Power Supply)

| Component | Specification |
|---|---|
| **UPS** | **APC Smart-UPS SRT 3000VA** |
| Capacity | 3000VA / 2700W |
| Runtime at Load | ~15 minutes at full load |
| Type | Online Double-Conversion |

**Why UPS is critical:**
- Protects active trades during power outages
- Prevents data corruption during model training
- Online double-conversion provides cleanest power for sensitive components

---

## TIER 2: HIGH-PERFORMANCE BUILD (Optimized Budget)

**Estimated Cost: $12,000 - $18,000**

Best performance-per-dollar for serious AI/ML trading work.

### Component Summary

| Component | Specification | Est. Price |
|---|---|---|
| **CPU** | **AMD Threadripper 7980X** (64C/128T, 5.1GHz boost) | $4,999 |
| **Motherboard** | **ASUS TRX50-SAGE WIFI** (TRX50 chipset) | $1,100 |
| **GPUs** | **2x NVIDIA RTX 4090 24GB** | $3,600 |
| **RAM** | **256GB DDR5-5600 ECC** (8x 32GB) | $1,200 |
| **OS Drive** | Samsung 990 Pro 2TB NVMe | $180 |
| **AI/ML Drive** | Samsung 990 Pro 4TB NVMe | $350 |
| **Data Drive** | WD Black SN850X 4TB NVMe | $300 |
| **Backup** | Seagate Exos X20 20TB HDD | $350 |
| **PSU** | Corsair HX1500i 1500W 80+ Platinum | $350 |
| **Cooler** | Arctic Liquid Freezer III 360 (sTR5) | $150 |
| **Case** | Fractal Design Torrent XL | $250 |
| **UPS** | CyberPower PR2200LCDRT2U | $800 |
| **Monitors** | 1x 34" Ultrawide + 2x 27" 4K | $1,200 |
| **Total** | | **~$14,800** |

### Key Differences from Tier 1

- 64 cores instead of 96 (still excellent for parallel processing)
- RTX 4090 instead of RTX 6000 Ada (24GB vs 48GB VRAM - use LoRA/QLoRA for LLM training)
- 256GB RAM instead of 512GB (sufficient with memory-mapped files and chunked processing)
- PCIe 5.0 NVMe but not enterprise-grade

---

## TIER 3: STRONG ENTRY BUILD (Budget-Conscious)

**Estimated Cost: $6,000 - $9,000**

Capable of all workloads with some compromises on training speed.

### Component Summary

| Component | Specification | Est. Price |
|---|---|---|
| **CPU** | **AMD Ryzen 9 9950X** (16C/32T, 5.7GHz boost) | $550 |
| **Motherboard** | **ASUS ProArt X870E-Creator WiFi** (AM5) | $500 |
| **GPU** | **1x NVIDIA RTX 4090 24GB** | $1,800 |
| **RAM** | **128GB DDR5-6000** (2x 64GB) | $350 |
| **OS Drive** | Samsung 990 Pro 2TB NVMe | $180 |
| **AI/ML + Data Drive** | WD Black SN850X 4TB NVMe | $300 |
| **Backup** | Seagate Barracuda 8TB HDD | $130 |
| **PSU** | Corsair RM1000e 1000W 80+ Gold | $170 |
| **Cooler** | Noctua NH-D15 G2 | $110 |
| **Case** | Fractal Design Torrent | $200 |
| **UPS** | CyberPower CP1500PFCLCD | $250 |
| **Monitors** | 1x 34" Ultrawide + 1x 27" 4K | $700 |
| **Total** | | **~$5,240** |

### Limitations at This Tier

- 16 cores limits parallel data preprocessing speed
- Single GPU limits training throughput (but still very capable)
- 128GB RAM requires chunked data loading for 10yr datasets
- Dual-channel DDR5 has less memory bandwidth than quad/octa-channel

---

## Software Stack Recommendations

### Operating System

| Software | Purpose |
|---|---|
| **Windows 11 Pro for Workstations** | NinjaTrader, OpenClaw (Windows-only apps) |
| **WSL2 + Ubuntu 24.04** | AI/ML development, Python, CUDA |
| **Docker Desktop** | Containerized ML pipelines |

### AI/ML Framework Stack

| Software | Purpose |
|---|---|
| **Python 3.11+** | Primary ML language |
| **PyTorch 2.x** | Deep learning framework (preferred for flexibility) |
| **CUDA 12.x + cuDNN 9.x** | GPU acceleration |
| **DeepSpeed** | Distributed training, ZeRO optimizer for LLMs |
| **Hugging Face Transformers** | LLM fine-tuning framework |
| **scikit-learn** | Classical ML models |
| **XGBoost / LightGBM** | Gradient boosting (GPU-accelerated) |
| **Pandas 2.x / Polars** | Data manipulation (Polars preferred for large tick data) |
| **Apache Arrow / Parquet** | Efficient tick data storage format |
| **DuckDB** | In-process analytics on large datasets |
| **Ray** | Distributed computing for hyperparameter tuning |
| **MLflow** | Experiment tracking and model registry |
| **Jupyter Lab** | Interactive development |

### Trading Software

| Software | Purpose |
|---|---|
| **NinjaTrader 8** | Charting, analysis, execution |
| **OpenClaw** | Multi-asset trading platform |
| **Custom Scanner** | Python-based stock scanner (built on top of data feeds) |
| **Redis** | Real-time data caching for scanner |
| **TimescaleDB** | Time-series database for tick data storage |

---

## Data Storage Architecture for 10 Years of Tick Data

### Data Size Estimates

| Data Type | Approximate Size (10 Years) |
|---|---|
| Single instrument, all ticks | 20-50 GB |
| ES (E-mini S&P 500) all ticks | 80-150 GB |
| 50 instruments, all ticks | 1-5 TB |
| 1000+ stocks daily bars + Level 2 | 2-8 TB |

### Recommended Data Format

```
Tick Data Pipeline:
  Raw CSV/Binary --> Apache Parquet (columnar, compressed)
                      --> Partitioned by: instrument/year/month
                      --> Compression: Zstandard (zstd)
                      --> Typical compression ratio: 5-10x

Storage Hierarchy:
  Hot (NVMe SSD):  Active training datasets, recent data
  Warm (SATA SSD): Preprocessed archives, older years
  Cold (HDD):      Raw backups, model checkpoint archives
```

### Optimized Data Loading for Training

```python
# Example: Efficient tick data loading with Polars + Parquet
import polars as pl

# Lazy evaluation - only loads what's needed
tick_data = (
    pl.scan_parquet("data/ticks/ES/**/*.parquet")
    .filter(pl.col("timestamp").is_between("2015-01-01", "2025-01-01"))
    .select(["timestamp", "price", "volume", "bid", "ask"])
    .with_columns([
        pl.col("price").rolling_mean(window_size=100).alias("sma_100"),
        pl.col("volume").rolling_sum(window_size=50).alias("vol_50"),
    ])
    .collect(streaming=True)  # Stream processing for memory efficiency
)
```

---

## Network Architecture for Low-Latency Trading

```
Internet (ISP 1 - Primary)
    |
    +-- Ubiquiti UDM Pro (Router/Firewall)
    |       |
    |       +-- VLAN 10: Trading (isolated, QoS priority)
    |       |       +-- Workstation NIC 1 (10GbE)
    |       |
    |       +-- VLAN 20: General
    |               +-- Workstation NIC 2 (1GbE)
    |
Internet (ISP 2 - Failover)
    |
    +-- Connected to UDM Pro WAN2 (automatic failover)
```

**Key networking principles:**
- **Dual ISP** with automatic failover (critical for live trading)
- **VLAN isolation** keeps trading traffic separate from general browsing/downloads
- **QoS rules** prioritize trading data over ML training data transfers
- **Wired connections only** - never use WiFi for trading

---

## Performance Benchmarks (Expected)

### Tier 1 Build - Expected Training Times

| Task | Estimated Time |
|---|---|
| Load 10yr tick data (single instrument, Parquet) | 15-30 seconds |
| Train XGBoost on 10yr tick features (100M rows) | 5-15 minutes |
| Train LSTM model (10yr, 1M sequences, 100 epochs) | 2-6 hours |
| Train Transformer model (10yr, 1M sequences, 100 epochs) | 4-12 hours |
| Fine-tune 7B LLM (LoRA, 10yr text-formatted data) | 8-24 hours |
| Fine-tune 13B LLM (QLoRA, 10yr text-formatted data) | 12-48 hours |
| Stock scanner (scan 5000 stocks real-time) | < 1 second per cycle |

### Simultaneous Workload Capability

With the Tier 1 build, you can run **all of the following simultaneously**:

- NinjaTrader with 20+ charts open
- OpenClaw with active trading
- Stock scanner processing 5000+ symbols
- Auto-execution engine
- 1x deep learning model training on GPU 1
- 1x ML model backtesting on CPU

---

## Build Checklist

### Pre-Build

- [ ] Verify motherboard BIOS supports your CPU revision
- [ ] Confirm PSU has enough PCIe power connectors for all GPUs
- [ ] Purchase thermal paste (Thermal Grizzly Kryonaut)
- [ ] Get anti-static wrist strap and mat
- [ ] Ensure case fits E-ATX motherboard (if applicable)

### Post-Build

- [ ] Install Windows 11 Pro for Workstations
- [ ] Enable XMP/EXPO profile in BIOS for DDR5 speed
- [ ] Set PCIe slots to Gen 5 in BIOS
- [ ] Install NVIDIA Studio drivers (more stable for compute)
- [ ] Install CUDA Toolkit 12.x
- [ ] Install WSL2 + Ubuntu 24.04
- [ ] Configure Docker Desktop with GPU passthrough
- [ ] Set up Python environment with conda/mamba
- [ ] Install PyTorch with CUDA support
- [ ] Configure NinjaTrader connection
- [ ] Set up dual ISP failover
- [ ] Configure UPS auto-shutdown via USB
- [ ] Set Windows power plan to "Ultimate Performance"
- [ ] Disable Windows Update auto-restart during trading hours
- [ ] Set up automated backups for tick data

### BIOS Optimizations

- [ ] Enable Precision Boost Overdrive (PBO) for AMD CPUs
- [ ] Set memory to rated XMP/EXPO speed
- [ ] Enable Resizable BAR (ReBAR) for GPU performance
- [ ] Disable C-States for lowest latency (if not concerned about idle power)
- [ ] Enable IOMMU for GPU passthrough capability

---

## Summary Recommendation

**For your specific use case (all workloads simultaneously), I strongly recommend TIER 1** with the following priority order for budget allocation:

1. **GPU (Highest Priority):** LLM training and deep learning are GPU-bound. Get the most VRAM you can afford. 2x RTX 6000 Ada (48GB each) is ideal.
2. **RAM (Second Priority):** 10 years of tick data loaded in memory dramatically speeds up all ML/DL workflows. 512GB minimum recommended.
3. **CPU (Third Priority):** Threadripper PRO for maximum PCIe lanes (multi-GPU) and cores (data preprocessing).
4. **Storage (Fourth Priority):** PCIe 5.0 NVMe for training I/O. Separate drives for OS, training, and data.
5. **Network (Fifth Priority):** Dual ISP with failover. 10GbE internal networking.
6. **UPS (Essential):** Non-negotiable for live trading with auto-execution.

---

*Document Version: 1.0*
*Last Updated: February 2026*
*Purpose: AI/ML Trading Workstation Configuration Guide*
