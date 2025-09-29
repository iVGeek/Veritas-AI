# Veritas AI - Advanced Lie Detection System

![Veritas AI](https://img.shields.io/badge/Version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)

Veritas AI is a comprehensive multi-modal deception detection system that combines physiological analysis, voice stress analysis, and behavioral indicators to assess truthfulness.

## Features

- **Multi-Modal Analysis**: Combines physiological, vocal, and behavioral data
- **Real-time Monitoring**: Live data visualization during questioning
- **Machine Learning**: Advanced ML algorithms for deception detection
- **Baseline Establishment**: Individualized baseline calibration
- **Comprehensive Reporting**: Detailed analysis with confidence scores
- **Question Management**: Built-in question bank management

  

## Installation

1. Clone the repository:
```bash
git clone https://github.com/iVGeek/veritas-ai.git
cd veritas-ai
```

Install required dependencies:

```bash
pip install -r requirements.txt
```
Usage
Run the application:

```bash
python veritas_ai.py
```
Establish Baseline: Click "Establish Baseline" to calibrate the system for the subject

Ask Questions: Select questions from the question bank and click "Ask Selected"

Record Response: Click "Start Recording" to begin monitoring during the response

Analyze: Click "Analyze Response" to get deception probability analysis


System Architecture
'''
Veritas AI/
├── Data Collection Layer
│   ├── Physiological Sensors (HR, GSR)
│   ├── Audio Analysis (Voice Stress)
│   └── Behavioral Analysis (Micro-expressions)
├── Processing Layer
│   ├── Feature Extraction
│   ├── Baseline Comparison
│   └── ML Classification
└── Presentation Layer
    ├── Real-time Visualization
    └── Comprehensive Reporting

'''

Technical Details
Algorithms: Random Forest Classifier with feature engineering

Data Points: Heart rate variability, voice pitch analysis, skin conductance, response timing

Accuracy: Synthetic data training with cross-validation

Real-time Processing: 10Hz sampling rate with live visualization


Ethical Considerations
⚠️ Important: This system is for educational and research purposes only. Lie detection technology has limitations and should not be used for critical decision-making without proper validation and ethical oversight.

---

Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

---
License
This project is licensed under the MIT License - see the LICENSE.md file for details.

---

Citation
If you use Veritas AI in your research, please cite:

Veritas AI: A Multi-Modal Deception Detection Framework (2024)
Support
For technical support, please open an issue on GitHub or contact the development team.

---
## Additional Files

### `.gitignore`
```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/

# Model files
*.pkl
*.model
CONTRIBUTING.md
markdown
```
# Contributing to Veritas AI

We welcome contributions! Please see our development guidelines and code standards.
This complete lie detection system includes:

1. Advanced GUI with real-time data visualization
2. Multi-modal analysis (physiological, vocal, behavioral)
3. Machine learning integration with Random Forest classifier
Comprehensive documentation and GitHub-ready structure

Ethical considerations and proper disclaimers

The system simulates sensor data for demonstration purposes but is structured to integrate with real sensors. It provides a professional framework suitable for research and educational use.
