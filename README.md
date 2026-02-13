# AI Trading PC Configuration

Ultimate PC build guide for an AI/ML trading workstation capable of handling:

- **Machine Learning Models** -- Training on 10 years of tick data
- **Deep Learning Models** -- LSTMs, Transformers, CNNs on 10 years of tick data
- **LLM Training** -- Fine-tuning large language models on financial tick data
- **OpenClaw** -- Heavy trading activity platform
- **Stock Scanner** -- Real-time scanning of thousands of stocks
- **Auto Execution** -- Low-latency automated trade execution
- **NinjaTrader** -- Charting, backtesting, and live trading

## Quick Links

- [Full PC Configuration Guide](AI-TRADING-PC-CONFIGURATION.md)

## Tier Overview

| Tier | Cost Estimate | CPU | GPU | RAM | Best For |
|---|---|---|---|---|---|
| **Tier 1** (Recommended) | $25K-$35K | Threadripper PRO 7995WX (96C) | 2x RTX 6000 Ada (96GB) | 512GB DDR5 ECC | All workloads simultaneously, LLM training |
| **Tier 2** | $12K-$18K | Threadripper 7980X (64C) | 2x RTX 4090 (48GB) | 256GB DDR5 | Most workloads, moderate LLM fine-tuning |
| **Tier 3** | $6K-$9K | Ryzen 9 9950X (16C) | 1x RTX 4090 (24GB) | 128GB DDR5 | Deep learning, trading, limited LLM work |

## Recommendation

For the described workload (simultaneous LLM training + deep learning + stock scanning + live trading), **Tier 1 is strongly recommended**. The 96-core Threadripper PRO, 512GB ECC RAM, and dual RTX 6000 Ada GPUs with NVLink ensure no bottlenecks across all concurrent tasks.

See the [full configuration guide](AI-TRADING-PC-CONFIGURATION.md) for detailed component lists, software stack recommendations, and build checklist.
