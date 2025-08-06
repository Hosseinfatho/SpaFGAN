# 🧬 SpaFGAN: Spatial Feature Graph Attention Network

> **Advanced spatial biology analysis using Graph Attention Networks for cellular interaction discovery**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen)](https://hosseinfatho.github.io/SpaFGAN/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org)

## 🚀 Quick Start

### Live Demo
**🌐 [Try it online](https://hosseinfatho.github.io/SpaFGAN/)**

### Local Development
```bash
# Backend (Python)
cd backend
conda create -n SPA python=3.8
conda activate SPA
pip install -r requirements.txt
python server.py

# Frontend (React)
cd frontend
npm install
npm run dev
```

## 🎯 What it does

SpaFGAN analyzes spatial biological data to discover cellular interactions using:

- **Graph Attention Networks (GAT)** for pattern recognition
- **ROI Analysis** with scoring system
- **Interactive Visualization** with Vitessce
- **Multi-marker Analysis** for cellular interactions

## 🔬 Key Features

- **Interactive ROI Selection** - Navigate through regions of interest
- **Cellular Interaction Analysis** - B-cell infiltration, T-cell maturation, etc.
- **Real-time Heatmaps** - Visualize marker interactions
- **Score-based Filtering** - Find high-scoring regions automatically
- **Multi-dimensional Data** - Handle complex spatial datasets

## 📊 Supported Interactions

- **B-cell infiltration**
- **T-cell maturation** 
- **Inflammatory zones**
- **Oxidative stress regulation**

## 🛠️ Tech Stack

- **Backend**: Python, PyTorch, Graph Attention Networks
- **Frontend**: React, Vitessce, D3.js
- **Data**: OME-Zarr format
- **Deployment**: GitHub Pages

## 📁 Project Structure

```
SpaFGAN/
├── backend/          # Python processing pipeline
├── frontend/         # React visualization app
└── models/           # Trained GAT models
```

---

**🔬 Built for spatial biology research | 🚀 Deployed on GitHub Pages**
