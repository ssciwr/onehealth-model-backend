Aedes albopictus model

###########################

This folder contains Octave scripts to run the Aedes albopictus model driven by climate and human population data.
In addition, there are example netCDF files for minimum and maximum temperature, precipitation and human population density for a test run.

###########################

Prerequisites

The model was tested for Octave v4.2.1.
It requires the following Octave packages pre-installed:

    netcdf
	ncarray

###########################

Usage

Infiles for climate and human population data can be specified in aedes_albopictus_model.m. These are currently the example files.
Running the script aedes_albopictus_model.m starts the model simulations for as many years specified.

Test

Running the script with default settings and example data as infiles should result in the plot example.png.

###########################
