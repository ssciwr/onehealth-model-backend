## Code to calculate and aggregate daily vectorial capcity data (old formula) to monthly data and 
## convert in format in which is suitable for old absolute R0 model calculation

## The output matrix can be directly use din absolute R0 code along with abundance matrix

### Input of this program will be min and max temperature and then we calculate vectorial capcity and then convert to desired format

# Task to do

# set this program to also run for diurnal temperature range by dividing a day into 100 time steps like mosquito abundance

### load libraries
### load libraries
library(ncdf4)
library(ggplot2)
library(terra)
library(leaflet)
library(lubridate)
library(terra)
library(raster)
library(terra)
library(reshape2)
library(dplyr)
library(lubridate)
library(exactextractr)
library(stringr)
library(sf)
library(xts)
library(zoo)
library(tidyverse)
library(viridis)



### put whole computation process in function for loop calculation of many years

process_temperature_data1 <- function(data1) {
  
  # Read the temperature data from the files
  #tmin_ <- ncvar_get(tmin, "t2m")  # Replace with the actual variable name
  #tmax_ <- ncvar_get(tmax, "t2m")  # Replace with the actual variable name
  
  # Close the NetCDF files when done
  #nc_close(tmin)
  #nc_close(tmax)
  
  tmean1 = ncvar_get(tmean, "t2m")  ### how to save the output tmean as nc file 
  
  ##calculate vc
  
  EIR = function(t){
    result<-1.04e-04 * t *(t - 10.39) * (43.05 - t)^(0.5)
    result <- ifelse(result < 0, 0, result)
    return(result)
  }               ### EXTRINSIC INCUBATION RATE
  
  
  lf <- function(t) {
    result <- 1.43 * (t - 13.41) * (31.51 - t)
    result <- ifelse(result < 0, 0, result)
    return(result)
  }              ## LIFESPAN
  
  
  a <- function(t) {
    result <- 1.93e-04 * t * (t - 10.25) * (38.32 - t)^(0.5)
    result <- ifelse(result < 0, 0, result)
    result[is.nan(result)] <- 0
    result[is.na(result)] <- 0
    return(result)
  }          ## biting rate
  
  c = function(t){
    result <- 4.39e-04 * t *(t - 3.62) * (36.82 - t)^(0.5)
    result <- ifelse(result < 0, 0, result)
    result[is.nan(result)] <- 0
    result[is.na(result)] <- 0
    return(result)
  }### vector competence
  
  
  vc=function(t) {
    ec = 0.000001
    k = a(t)*a(t)*c(t)*exp(-(1/(lf(t)+ec))*(1/(EIR(t)+ec))) * lf(t)        ### mortality rate = inverse of lifespan
    k[is.nan(k)] <- 0
    k[is.na(k)] <- 0
    return(k)}
  
  m = vc(tmean1)
## Now vc(tmean) is a 3D matrix and we need it to good format for abundace calculation convert and save it to an .nc file for later work

## The first task is to aggregated 3d daily data to monthly and bring it to in 2d format matrix with row representing the lat long combination and column as monthly aggregated data

## Now convert the abundance also to montly aggregation 



# You can replace this with your actual data loading and processing logic

  # Your existing processing code here
  # Extract latitude, longitude, and vc data from earlier formula
  lat <- ncvar_get(tmean, "latitude")
  lon <- ncvar_get(tmean, "longitude")
  vectorial_cap <- m  
  
  # assigning row names as longitude and column as latitude
  colnames(vectorial_cap)<-lat
  rownames(vectorial_cap)<-lon
  
  
  temp_long<-melt(vectorial_cap)
  
  t = ncvar_get(tmean, "time")     # since vectorial capcity is using the nc data of tmax, so the time of vc will be same as time of nc file of tmax. We can't use vc beacuse it is now a variable matrix not netcdf file 
  
  #time unit: hours since 1900-01-01, check the origin of file 
  ncatt_get(tmean,'time')
  
  
  # if days since origin is given, each input file have origin changes every year 
  #timestamp = as_datetime(c(t*60*60*24),origin = "2022-01-01")    # change the origin for each loop according to year
  
  # Assuming t is the time variable from the netCDF file
  timestamp <- as.POSIXct(t * 60 * 60 * 24, origin = paste0(year,"-01-01"), tz = "UTC")  # change the origin for each loop according to year
  
  # Now 'timestamp' contains the converted time values
  
  # if Hours since origin is given
  #timestamp = as_datetime(c(t*60*60),origin="0001-01-01")
  
  
  # checking the months covered in complete time frame
  times = month(timestamp)
  times
  
  #nc_close(tx)
  
  
  #use timestamps as names in temperature raster file
  #names(abundr) = timestamp # This line is just to give a better look to raster plot
  
  time = rep(timestamp, each = length(lat) * length(lon))   # to give each day of the data corresponding date in dd my yyyy format
  
  temp_long$timen = time           # adding the corresponding date column to melted data frame
  
  # Convert the 'timen' column to a Date format
  temp_long$timen <- as.Date(temp_long$timen)
  
  # Extract month and year from the 'timen' column
  temp_long <- temp_long %>%
    mutate(year_month = format(timen, "%Y-%m"))
  
  #temp_long <- temp_long %>%
  #  mutate(day = format(timen, "%d"))
  
  #temp_long <- temp_long %>%
  #  mutate(year_month_day = format(timen, "%Y-%m-%d"))
  
  # Aggregate data by month for each latitude-longitude combination
  result <- temp_long %>%
    group_by(year_month, Var2, Var1) %>%
    summarise(mean_value = mean(value))      # grouping by the each month 
  
  # Returning the aggregated result
  return(result)
}



#setwd("/Users/pratik/Desktop/ae_albopictus_model")
setwd("/Users/pratik/Desktop/Collab_data/Zia_collab/Climate_data_USA/temperature_mean_USA")

temp = nc_open("era5land_tmean_USA_daily_0.25_2019.nc")  # A nc file is required of same dimension to set rows of matrix, any input file will be ok, for only once

lat <- ncvar_get(temp, "latitude")
lon <- ncvar_get(temp, "longitude")

# assigning row names as longitude and column as latitude
#colnames(temp)<-lat
#rownames(temp)<-lon

df = expand.grid(x=lon,y=lat) # to get the number of rows for the next step


# Initialize an empty matrix to store aggregated values (final values) with the help of 
final_matrix_vc <- matrix(nrow = length(df$x), ncol = 0)     # instead of this we can also swap the first two columns of output of function


# path and file name, set dname
ncpath <- "/Users/pratik/Desktop/Collab_data/Zia_collab/Climate_data_USA/temperature_mean_USA/"   # location of nc file
ncname1 <- "era5land_tmean_USA_daily_0.25_" 
#ncname2 <- "era5land_tmean_EU_daily_0.25_" 


#!! don't forget to initialize final matrix as above each time if you are changing the years in loop

# Loop through each year (assuming you have data for multiple years)
for (year in 1995:2019) {  # Replace "year1" and "year2" with your actual year values
  
  tmean <- nc_open(paste(ncpath, ncname1, year, ".nc", sep=""))  # read the file
  #tmean <- nc_open(paste(ncpath, ncname2, year, ".nc", sep=""))  # read the file
  
  # Process temperature data and get aggregated result
  result <- process_temperature_data1(tmean)  # it doesnot matter whether tmin is first arguement or tmax is first
  
  # Pivot the result to wide format
  wide_result <- pivot_wider(result, names_from = year_month, values_from = mean_value)
  
  # Add the wide_result as additional columns to the final matrix
  final_matrix_vc <- cbind(final_matrix_vc, as.matrix(wide_result[, -c(1:2)]))  # delete the first two lat long columns to get the abundance
  
  cat("Aggregation of", year, "completed\n")
}


# Checking the aggregated montly data
#temp23 = rast("Mosquito_abundance_EU_2012.nc")

df = expand.grid(x=lon,y=lat)
rast_R0_Annual1 = rasterFromXYZ(data.frame(df,final_matrix_vc[,296]))
proj4string(rast_R0_Annual1)<-CRS("EPSG:4326")
plot(rast_R0_Annual1)

# Save the final matrix to a CSV file
write.csv(final_matrix_vc, "vc_monthly_file_.csv", row.names = FALSE)
