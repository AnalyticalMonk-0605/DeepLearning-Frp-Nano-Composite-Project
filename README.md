# Deep Learning–Driven FRP Nano-Composite Estimator

## Project context

This repository contains the source code and assets for the final year B.E. Computer Science and Engineering project:

“Deep Learning-Driven Prediction and Optimization of Mechanical Properties in Nano-Filler Reinforced FRP Woven Composites”.

The work was carried out at AAA College of Engineering and Technology, Sivakasi under the supervision of Dr. J. Hemalatha, M.E., Ph.D., as a bonafide academic project by:

- G Sanjay (Reg. No. 953722104045)  
- Sutakar S J (Reg. No. 953722104051)

This repository is intended primarily for project evaluation, academic review, and personal archival.

## Technical overview

The project develops an end-to-end software system that:

- Uses a Gated Recurrent Unit (GRU) deep learning model to predict tensile and flexural stress of nano-silica reinforced glass–epoxy FRP woven composites over 0–25 wt% nano-silica.  
- Extends the base experimental dataset to cover both the reinforcement-dominated (0–15%) and agglomeration-dominated (15–25%) regimes, capturing non-monotonic behaviour.  
- Integrates an Application Simulation Engine that translates predicted properties into engineering metrics such as safety factors, pressure capacity, service temperature, manufacturing cost, and suitability labels for six industrial scenarios (automotive, bridge, wind, marine, drone, chemical equipment).  
- Exposes the workflow through an interactive Streamlit web dashboard, so users can explore “what-if” scenarios without writing code.

## Core components

The system is organised in four conceptual layers that correspond to Python modules in this repository:

- **Data layer**  
  - `frpfullyextended.csv`: Extended experimental dataset (nano-silica %, specimen ID, tensile stress, flexural stress).  
  - `frprnnmodeloptimized.h5`: Trained GRU model for strength prediction.

- **Prediction layer**  
  - `frppredictor.py`: Loads the dataset and GRU model, constructs sliding-window sequences, applies scaling, and predicts tensile and flexural stress for a user-specified nano-silica percentage.

- **Simulation layer**  
  - `advancedestimator.py`: Implements the Application Simulation Engine. From the predicted strengths plus user inputs (panel thickness, area, operating temperature, annual volume), it computes:  
    - reliability-adjusted allowable stresses  
    - safety factors for tensile, flexural, and pressure  
    - pressure capacity  
    - service temperature estimate  
    - component mass and manufacturing cost  
    - overall suitability score and label (Excellent, Viable, Conditional, Redesign)  
    - ranking across six application scenarios.

- **Presentation layer**  
  - `app.py`: Streamlit dashboard with tabs for scenario simulation, scenario ranking, model metrics, training data, and engineering assumptions.

Auxiliary scripts such as `rnnupdated1525percent.py` (training) and `modelevaluate.py` (offline evaluation) reproduce model training and validation workflows.

## Model and performance

- Architecture: Two-layer GRU network (256 and 128 units) with dense layers and a 2-neuron output for tensile and flexural stress.  
- Training: Sliding-window sequence construction, 80/20 train–test split, StandardScaler normalisation, early stopping, and learning rate reduction on plateau.  
- Implementation: TensorFlow/Keras, up to 500 epochs with typical convergence between 150–300 epochs.  
- Test performance (20% hold-out):  
  - R² ≈ 0.91 (tensile) and 0.89 (flexural)  
  - RMSE and MAE values suitable for engineering design decisions across 0–25 wt% nano-silica.

Detailed metrics and comparisons (actual vs predicted, residuals, regression plots) are saved in `frprnnmodelresults.xlsx` and visualised inside the Streamlit dashboard.

## Application Simulation Engine

For each of the six scenarios (automotive body panel, bridge strengthening laminate, wind turbine shell skin, marine deck overlay, drone fairing/radome, chemical equipment access panel), the engine uses fixed reference parameters and:

1. Applies reliability adjustment based on distance from 15 wt% nano-silica.  
2. Reduces raw strengths using scenario-specific utilisation factors.  
3. Applies thickness and area corrections for panel geometry.  
4. Calculates pressure capacity in kPa.  
5. Computes separate safety factors for tensile, flexural, and pressure, and chooses the governing factor.  
6. Estimates service temperature from nano-silica content and scenario environment.  
7. Estimates manufacturing cost, including nano-silica premium, processing penalty, waste, and volume discount.  
8. Produces a suitability score and label, and ranks all six scenarios.

## Academic usage and citation

If this work is referenced in academic or technical contexts, please cite the base experimental work and this project appropriately.

- Nagendran M. et al., 2024 – base experimental study on nano-silica reinforced glass–epoxy FRP.  
- G. Sanjay, S. J. Sutakar, 2026 – “Deep Learning-Driven Prediction and Optimization of Mechanical Properties in Nano-Filler Reinforced FRP Woven Composites”, B.E. Project, AAA College of Engineering and Technology, Sivakasi.

## License and permissions

This repository is **not** a general open-source library. It represents an academic final year project with significant original modelling, simulation, and software engineering work.

- **No free/open use**  
  The code, datasets, trained models, and documentation in this repository must not be reused, modified, or integrated into other projects (academic, commercial, or open source) without explicit written permission from the authors.

- **Collaboration and reuse**  
  Any form of collaboration, reuse, derivative work, or redistribution requires prior written approval. For research, coursework, or industrial use, contact the authors and obtain consent before using any part of this repository.

See the `LICENSE` file in this repository for the full legal text.

## Contact

For permissions, collaboration requests, or technical questions:

- Author: G Sanjay   
- Email: sanjaygram0605@gmail.com  
- GitHub: https://github.com/AnalyticalMonk-0605
