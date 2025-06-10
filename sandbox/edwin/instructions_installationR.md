# Installation of Packages for R

## NetCDF
  1. Install NetCDF C library (and development files)
  ```bash
    sudo apt update
    sudo apt install libnetcdf-dev
    sudo apt install netcdf-bin
  ``` 
  2. Install the `netcdf4` R package
  ```r
    install.packages("ncdf4")
  ```

## Terra
  1. Install Terra dependencies
  ```bash
    sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
    sudo apt-get update
    sudo apt-get install libgdal-dev libgeos-dev libproj-dev libnetcdf-dev libsqlite3-dev libtbb-dev
  ``` 
  2. Install the `netcdf4` R package
  ```r
    remotes::install_github("rspatial/terra")
  ```

## sf
  1. Install fortran
    ```bash
    sudo apt update
    sudo apt install gfortran
    ```
  2. Install sf dependencies
  ```bash
    # Add the ubuntugis-unstable PPA (recommended for up-to-date geospatial libraries)
    sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
    sudo apt update
    sudo apt upgrade # Important to upgrade existing geospatial libraries from this PPA

    # Install the required development libraries
    sudo apt install libudunits2-dev libgdal-dev libgeos-dev libproj-dev libsqlite3-dev build-essential
  ``` 
  3. Install the `netcdf4` R package
  ```r
    remotes::install_github("rspatial/terra")
  ```
