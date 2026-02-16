
# Model Card: WNV-R0

## Overview
**Model Name:** WNV-R0  
**Version:** 1.0  
**Type:** Epidemiological (Vector–Host Compartmental Model)  
**Task:** Basic Reproduction Number (R₀) Estimation  
**Domain:** West Nile Virus Transmission  

**Organization:** HeiPlanet  
**Date Created:** 2026-01-12  
**Contact:** your-email@organization.com  

## Description
WNV-R0 estimates the basic reproduction number (R₀) for West Nile Virus transmission
using environmental, entomological, and host population data. The model is intended
for seasonal and regional risk assessment rather than real-time forecasting.

## Intended Use
### In Scope
- Seasonal WNV risk assessment
- Regional comparison of transmission potential
- Climate and vector scenario analysis
- Public health preparedness planning

### Out of Scope
- Individual risk prediction
- Real-time outbreak detection
- Automated policy decisions

## Architecture
- Deterministic compartmental vector–host model
- Temperature-dependent transmission parameters
- Environmental and seasonal forcing

## Inputs
| Feature | Description | Units |
|-------|------------|-------|
| Temperature | Daily mean temperature | °C |
| Precipitation | Daily precipitation | mm |
| Vector Density | Mosquito abundance index | Index |
| Host Population | Susceptible hosts | Count |
| Seasonal Factors | Photoperiod/seasonality | Encoded |

## Output
- **R₀ Estimate:** Dimensionless reproduction number

## Training Data
- CDC WNV Surveillance Data
- ERA5 Climate Reanalysis
- Entomological surveys  

_Preprocessing includes missing value imputation, rate transformations, and seasonal normalization._

## Evaluation
**Metrics:** RMSE, MAE, Correlation Coefficient (TBD)  
**Testing Data:** Historical WNV outbreak records

## Assumptions
- Stable seasonal transmission dynamics
- Homogeneous spatial mixing
- No explicit modeling of interventions

## Limitations
- Data sparsity in some regions
- Deterministic structure
- Simplified biology and host dynamics
- Seasonal (non–real-time) design

## Bias & Fairness
- Biased toward regions with better surveillance
- Outputs should not be equated with health system capacity

## Potential Harms
- Misuse without expert review
- Resource misallocation

## Risk Mitigation
- Epidemiologist review
- Uncertainty communication
- Regular recalibration

## Maintenance
- Annual updates or as new data become available
- Semantic versioning
- Two-year deprecation window

## Datasets
- CDC West Nile Virus Activity
- Copernicus ERA5 Climate Data

## Publications & Related Models
None currently listed.
