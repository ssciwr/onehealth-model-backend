## Code to aggregate daily abundance data to monthly data and convert in format in which is suitable for old absolute R0 model calculation

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
library(sf)
library(exactextractr)
library(tidyr)
## Now convert the abundance also to montly aggregation 



#abund = nc_open("Mosquito_abundance_EU_2010.nc")
#abundr= rast("Mosquito_abundance_EU_2010.nc")
#abund1 = ncvar_get(abun, "adults")





# Pivot the data to wide format with each month as a column
#wide_result <- pivot_wider(result, names_from = year_month, values_from = mean_value)

# If you want a matrix, you can extract the numeric values
#result_matrix <- as.matrix(wide_result[, -c(1:2)])




# You can replace this with your actual data loading and processing logic
process_temperature_data <- function(data) {
  # Your existing processing code here
  # Extract latitude, longitude, and data
  lat <- ncvar_get(abund, "latitude")
  lon <- ncvar_get(abund, "longitude")
  abundance <- ncvar_get(abund, "adults")
  
  # assigning row names as longitude and column as latitude
  colnames(abundance)<-lat
  rownames(abundance)<-lon
  
  
  temp_long<-melt(abundance)
  
  t = ncvar_get(abund, "time")
  
  #time unit: hours since 1900-01-01, check the origin of file 
  ncatt_get(abund,'time')
  
  
  # if days since origin is given, each input file have origin changes every year 
  #timestamp = as_datetime(c(t*60*60*24),origin = paste0(year,"-01-01"))    # change the origin for each loop according to year
  
  # sometimes above function may have some problem, so use below function
  timestamp <- as.POSIXct(t * 60 * 60 * 24, origin = paste0(year,"-01-01"), tz = "UTC")  # change the origin for each loop according to year
  
  # if Hours since origin (sometime fixed to a specific year, sometimes variable dependending on nc input file in abundance) is given
  #timestamp = as_datetime(c(t*60*60),origin="0001-01-01")  # change year in origin according to time unit (for example hours since when)
  
  
  # checking the months covered in complete time frame
  times = month(timestamp)
  times
  
  #nc_close(tx)
  
  
  #use timestamps as names in temperature raster file
  #names(abundr) = timestamp # This line is just to give a better look to raster plot
  
  time = rep(timestamp, each = length(lat) * length(lon))    # to give each day of the data corresponding date in dd my yyyy format
  
  temp_long$timen = time            # adding the corresponding date columnn to melted data frame
  
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
final_matrix_abun <- matrix(nrow = length(df$x), ncol = 0)    # instead of this we can also swap the first two columns of output of function


# path and file name, set dname
#ncpath <- "/Users/pratik/Desktop/ae_albopictus_model/"   # location of nc file
ncpath <- "/Users/pratik/Desktop/Collab_data/Zia_collab/Mosquito_abundance_USA/"
ncname <- "Mosquito_abundance_USA_"  


## first check the time of abundance file whether its hours since origin or days since origin and then change the function above accordingly

# Loop through each year (assuming you have data for multiple years)
for (year in 1995:2019) {  # Replace "year1" and "year2" with your actual year values
  
  abund <- nc_open(paste(ncpath, ncname, year, ".nc", sep=""))  # read the file
  
  
  # Process temperature data and get aggregated result
  result <- process_temperature_data(abund)
  
  # Pivot the result to wide format
  wide_result <- pivot_wider(result, names_from = year_month, values_from = mean_value)
  
  # Add the wide_result as additional columns to the final matrix
  final_matrix_abun <- cbind(final_matrix_abun, as.matrix(wide_result[, -c(1:2)]))  # delete the first two lat long columns to get the abundance
  
  cat("Aggregation of", year, "completed\n")
}


# Save the final matrix to a CSV file
#write.csv(final_matrix_abun, "abun_monthly_file.csv", row.names = FALSE)





# Checking the aggregated montly data
#temp23 = rast("Mosquito_abundance_EU_2012.nc")

df = expand.grid(x=lon,y=lat)
rast_R0_Annual = rasterFromXYZ(data.frame(df,final_matrix_abun[,296]))
proj4string(rast_R0_Annual)<-CRS("EPSG:4326")
plot(rast_R0_Annual)

