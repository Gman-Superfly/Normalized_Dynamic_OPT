# NormalizedDynamics: Project Organization & Publication Strategy

## Current status

The NormalizedDynamics repository is now complete with comprehensive implementation, validation, and documentation. This document outlines the current organization and future publication strategy.

## Repository structure

### Core Algorithm Implementation
```
src/
├── normalized_dynamics_optimized.py     # Optimized algorithm with adaptive parameters
├── normalized_dynamics_smart_k.py       # Smart K parameter adaptation
├── smart_sampling.py                    # Biological sampling strategies
├── streaming_simulator.py               # Real-time data simulation
├── run_tests.py                        # Master test runner
└── __init__.py
```

### Comprehensive Test Suite
```
tests/
├── test_normalized_dynamics.py          # Core unit tests
├── test_comprehensive_visualizations.py # Visual validation
├── test_pancreas_endocrinogenesis.py   # Pancreatic development
├── test_biological_metrics.py           # Standard biological metrics
├── test_enhanced_biological_metrics.py  # DPT-based evaluation
├── test_synthetic_developmental.py      # Ground truth validation
├── test_gaia_data.py                   # Astronomical data
├── test_wine_dataset.py                # Chemical classification
├── test_mouse_brain_cortical.py        # Neural tissue analysis
├── biological_metrics.py               # Metrics implementation
├── enhanced_biological_metrics.py      # Advanced DPT metrics
├── synthetic_developmental_datasets.py # Data generators
└── smart_sampling_enhanced_analysis.py # Sampling analysis
```

### Web Application
```
app.py                                  # Flask web interface
templates/
├── index.html                          # Main landing page
├── pancreas_analysis.html              # scRNA-seq analysis
├── biological_metrics.html             # Biological validation
├── smart_sampling.html                 # Sampling strategies
├── gaia_analysis.html                  # Astronomical data
├── streaming_demo.html                 # Real-time demo
├── synthetic_developmental.html        # Synthetic biology
└── mouse_brain_cortical.html          # Brain tissue analysis

static/
├── css/style.css                       # Application styling
├── js/main.js                         # Interactive functionality
└── results/                           # 150+ generated figures
```

### Datasets
```
data/
├── Pancreas/endocrinogenesis_day15.h5ad # Key validation dataset
├── cortex_data.zip                      # Mouse brain data
├── gaia_data_*.csv                      # Gaia star catalogs
├── simple_wikipedia_processed.json      # Text embeddings
└── [Wine data via sklearn]             # No separate file needed
```

### Documentation
```
docs/
├── ALGO_SUMMARY.md                              # Algorithm summary
├── METHODOLOGY_TRANSPARENCY.md                   # Evaluation methodology
├── ND_OPT_OVERVIEW.md                            # Overview documentation
├── NormalizedDynamics_OG_Technical_Documentation_deprecated_FEB_2025.py # Algorithm specification
├── README_WRITEUP.md                             # Comprehensive documentation
├── repo_plans/                                   # Project planning documents
│   ├── PROJECT_ORGANIZATION_PLAN.md              # This file - repository structure
│   └── REPOSITORY_EXCELLENCE_SUMMARY.md          # Repository quality assessment
├── smart_sampling/                               # Smart sampling documentation
│   ├── SMART_SAMPLING.md                         # Sampling methodology
│   └── SMART_SAMPLING_RESULTS.md                 # Sampling analysis results
└── tests/                                        # Test documentation
    ├── README_tests.md                           # Test infrastructure guide
    └── TEST_SETUP_SUMMARY.md                     # Test setup procedures
```

### Root Directory Files
```
├── README.md                           # Main project documentation
├── LICENSE                             # MIT license
├── requirements.txt                    # All dependencies including astroquery/astropy
├── app.py                              # Flask web application
├── live_sensor_demo.py                 # Standalone streaming demonstration
├── download_gaia_data.py               # GAIA data acquisition utility
├── prepare_datasets.py                 # Dataset preprocessing scripts
├── pancreas_analysis_results.csv       # Quantitative biological validation results
├── wine_classification_results.csv     # Classification benchmark metrics
└── LLMS.txt                            # Comprehensive project overview
```

---



---

## Future research directions

### **Study 2: K-Independence Properties** (3-6 months)
**Focus**: Mathematical analysis of velocity field independence from K parameter

**Required Files**:
- `k_independence_tests.py` - Experimental framework
- `test_k_independence_rigorous.py` - Comprehensive testing  
- `ablation_study_plan.md` - Mechanism identification
- `velocity_study_results/` - Preliminary data

**Research Questions**:
- Why does velocity computation show K-independence?
- What are the mathematical implications?
- Connection to gauge theory/invariance principles?

### **Study 3: Dynamics** 
**Focus**: Emergence of information-theoretic principles

**Required Files**:
- `Information_Dynamics_and_Fundamental_Physics.md` - Theoretical framework
- `Information_Dynamics_Framework.md` - Mathematical foundations
- `theoretical_analysis/` - Deep mathematical analysis

**Research Questions**:
- How does Free Energy Principle emerge from the algorithm?
- Connections to physical constants and conservation laws?
- Implications for understanding computation in nature?

### **Study 4: Extended Applications** (Ongoing)
**Focus**: Domain-specific applications

**Planned Work**:
- Hi-C genomic interaction networks
- Spatial transcriptomics with tissue structure
- Time-series biological processes
- Network dynamics and evolution

---



### For Future Studies
1. **K-Independence Investigation**
   - Set up systematic ablation studies
   - Develop theoretical framework
   - Connect to known mathematical principles
   (known right now, K independence after x samples is related to stabilizations of the system)

2. **Free Energy Analysis not really needed, wrong direction**
   - Formalize mathematical connections
   - Develop sound proofs
   - Explore physical interpretations (meh)

---



---

## Repository Statistics

- **Core Algorithm**: ~850 lines (optimized implementation LOL)
- **Smart K Variant**: ~500 lines (adaptive version)
- **Test Suite**: 15+ comprehensive test files
- **Web Interface**: 8 interactive pages, 880+ lines
- **Visualizations**: 150+ comparison figures
- **Documentation**: 5 major documents
- **Datasets**: 4 domains (biology, astronomy, chemistry, synthetic)

---


---

*Keep it simple stupid* 