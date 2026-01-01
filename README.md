# Systematic Comparison of Prompt Strategies for SAM in Remote Sensing

This repository contains the code and resources for the paper:

**"A Systematic Comparison of Prompt Strategies for Segment Anything Model in Remote Sensing"**

## 📋 Overview

This study systematically compares three prompt strategies for the Segment Anything Model (SAM) in remote sensing applications:
1. **Center Point** - Single point at object centroid
2. **Bounding Box** - Tight rectangle around object
3. **Multiple Points** - Three points at object corners

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SAM-Prompt-Comparison.git
cd SAM-Prompt-Comparison

# Install dependencies
pip install -r requirements.txt

# Download SAM model
python -m src.download_sam
