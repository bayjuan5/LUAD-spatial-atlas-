# ROSIE: Spatial Mapping of Lung Adenocarcinoma Evolution

This repository contains the computational framework for **ROSIE**, a deep-learning system designed to decode the spatial immune landscape and architectural evolution of Lung Adenocarcinoma (LUAD) from standard H&E-stained images.

## 🚀 Overview
The evolutionary transition from preinvasive lesions to invasive cancer is often hidden in routine pathology. ROSIE utilizes a 50-channel Fully Convolutional Network (FCN) to infer multiplexed protein expressions, enabling high-resolution mapping of the tumor microenvironment (TME) across 114 stage-resolved specimens.

## 🛠️ Analytical Pipeline

### 1. Multiplex Signal Inference (`inference.py`)
Translates standard H&E Whole Slide Images (WSI) into 50-plex protein expression maps. 
- **Core Logic:** Weighted-blending sliding-window inference.
- **Output:** 50-channel OME-TIFF and cell-level signal JSONs.

### 2. Phenotypic Profiling (`phenotype_umap.py`)
Classifies cell types (Tregs, CD8+ T cells, B cells, etc.) using a fuzzy marker-logic (V9.0). 
- **Visualization:** Generates high-dimensional phenotypic landscapes via UMAP projections.

### 3. Spatial Topology & Niche Analysis (`spatial_clustering.py`)
Quantifies "Architectural Fragmentation" as described in our *Cancer* manuscript.
- **Method:** DBSCAN-based spatial clustering to identify localized cell niches and interaction patterns during malignant progression.

### 4. Progression Modeling (`dynamics_simulation.py`)
Simulates the transition trajectory (Normal → AAH → AIS → MIA → IAC) using a **Timed Petri Net (TPN)**.
- **Insights:** Models the immune-reprogramming cascade and state-transition rates.

## 📊 Key Findings
- Identified "Architectural Fragmentation" as a hallmark of early invasive transition.
- Delineated a conserved two-phase immune reprogramming cascade.
- Provided a scalable framework for identifying stage-specific microenvironmental vulnerabilities.

## 📜 Citation
If you find this work useful, please cite our manuscript:
> **Huang, B., & Zhu, B. Architectural Fragmentation and Immune Reprogramming Define Early Malignant Transition in Lung Adenocarcinoma. Laboratory Investigation (in revision).**

## ✉️ Contact
For questions regarding the ROSIE framework or data requests, please contact:
**Beibei Huang, Ph.D.** - [bbh@imdlab.net](mailto:bbh@imdlab.net)
