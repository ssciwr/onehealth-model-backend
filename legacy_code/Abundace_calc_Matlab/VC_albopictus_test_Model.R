### load libraries
library(ncdf4)
library(ggplot2)
library(terra)
library(leaflet)
library(lubridate)
library(terra)
library(raster)
library(sf)
library(exactextractr)


setwd("/Users/pratik/Downloads")
tmin = nc_open('cru_ts4.06.2021.2021.tmn.dat.nc')
tmax = nc_open('cru_ts4.06.2021.2021.tmx.dat.nc')

# Read the temperature data from the files
tmin_ <- ncvar_get(tmin, "tmn")  # Replace with the actual variable name
tmax_ <- ncvar_get(tmax, "tmx")  # Replace with the actual variable name

# Close the NetCDF files when done
nc_close(tmin)
nc_close(tmax)

tmean = (tmax_ + tmin_)/2 ### how to save the output tmean as nc file 



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
   k = a(t)*a(t)*c(t)*exp(-(1/(lf(t)+ec))*(1/(EIR(t)+ec))) * lf(t)*0.8        ### mortality rate = inverse of lifespan
   k[is.nan(k)] <- 0
   k[is.na(k)] <- 0
   return(k)}

m = vc(tmean)
## Now vc(tmean) is a 3D matrix and we need it to convert and save it to an .nc file for later work, we will 
## overwrite the tmin file to store tmean data as it have the same dimension and properties. 

################################################################################################################

## This part of code is just to look at unimodel nature of trait and vectorial capacity
#t <- 1:50

#par(mfrow=c(2,3))
#plot(EIR(t), ylab="Extrinsic incubation rate (1/days)", xlab="Temperature", type="l")
#plot(lf(t), ylab="lifespan", xlab="Temperature", type="l")
#plot(c(t), ylab="Vector competence", xlab="Temperature", type="l")
#plot(a(t), ylab="Biting rate", xlab="Temperature", type="l")
#plot(vc(t), ylab="Vectorial capacity", xlab="Temperature", type="l")

###################################################################################################

## So re writing the original nc file to have the vc data in same format
setwd("/Users/pratik/Downloads")

# Create the output file name
outfile <- paste0("vc_test_", "World2021_", ".nc") # name of new overwrite file

# Check if the output file already exists and delete it if it does
if (file.exists(outfile)) {
  file.remove(outfile)
}

# Copy the content of tmax to the output file
file.copy("cru_ts4.06.2021.2021.tmn.dat.nc", outfile)

# Open the output file for writing
ncid <- nc_open(outfile, write = TRUE)   # enalbling overwriting in the tmin file


k = ncvar_rename(ncid, "tmn", "vc")  ## replace original variable name with new variable name and make a new name

#variable = ncvar_get(k, "eggs") # to check whether varible is replaced

ncvar_put(k, "vc", m) # overwriting tmean data to original tmin data
nc_close(k)

##opening the rast version of vc file saved


a = rast("vc_test_World2021_.nc")
a                          ## only one variable named vc, overwriting the original variable of new nc file

R01 = nc_open("vc_test_World2021.nc")
R0 = ncvar_get(R01, "vc")

# Calculate the mean of the third layer
mean_third_layer <- mean(R0[,,3])

# Create a new 2D array with the mean
mean_matrix <- array(mean_third_layer, dim = c(dim(myArray)[1], dim(myArray)[2]))


lat_ga = ncvar_get(R01, "lat")
lon_ga =  ncvar_get(R01, "lon")

# assigning row names as longitude and column as latitude
colnames(R0)<-lat_ga
row.names(R0)<-lon_ga

# creating a data frame of all coordinates, with lat long combination
area_grid_long<-melt(R0)

# First, lets find unique combinations of Var1 and Var2

df = expand.grid(x=lon_ga,y=lat_ga)
# To check the that how monthly avaerage looks like (is it correct or not), whether lat long combinations are good with monthly data


r0_one = a[[1:12]]



annual_R0 = rowMeans(R0)
rast_R0_Annual = rasterFromXYZ(data.frame(unique_combinations,annual_R0))
proj4string(rast_R0_Annual)<-CRS("+init=epsg:4326")
plot(rast_R0_Annual)

#reading the shapefile
setwd("/Users/pratik/Downloads")
shp_file <- "NUTS_RG_01M_2021_3035.shp"

# Read the shapefile
shapefile1 <- st_read(shp_file)

shapefile <- st_transform(shapefile1, crs = "+proj=longlat +datum=WGS84")

#Cropping the data so that it matches with area covered by shapefile i.e. both have same lat long extension
r0_usa <- crop(r0_one, extent(shapefile))
#we can print r0_usa to check extent of latitude and longitude

#to assign same CRS to ro_usa as that of shapefile if they don't have same CRS
crs(r0_usa) <- crs(shapefile)

#if we want to know all the values of r0_usa with a region of shapefile we can use below command it will give all the
#r0_usa for a single geometry of shapefile. Note that ID number 1 corresponds to first row of shapefile, and if say ID
# number 6 has multiple values then all the values fall to 6th geometry of shapefile. The total number of ID's will
# be eqaul to rows of shapefi;e
#r0_values <- extract(r0_usa, shapefile, df = TRUE)

# to get R0 values based on weighatge area it covered within region this has rows equal to shapefile row 
# remeber to import library(exactextractr)
output_data = exact_extract(r0_usa,shapefile,
                            fun="weighted_mean",weights="area")

# Write otput data to CSV, if you want to see the data
#write.csv(output_data, file = "output_values.csv", row.names = FALSE)

# Combine shapefile details in a data frame
shp_to_df <- as.data.frame(shapefile)
req_df <- shp_to_df[,1:7] # only required information from shapefile not geometry

# Add the output data to the 'req_df' dataframe
final_df <- cbind(req_df, output_data)


# Add a new column with the total count of values greater than 1 in each row
final_df$total_count_greater_than_1 <- apply(final_df[, 8:ncol(final_df)], 1, function(row) sum(row > 1))



# Print the updated data frame
print(final_df[,20])


# Join the shapefile with the output_data based on NUTS_ID
kk_df <- merge(shapefile, final_df, by.x = "COUNTYFP", by.y = "COUNTYFP")

# Define your custom color palette with three colors
custom_colors <- c("blue","green")


# Plot the shapefile with output_data using a continuous color scale
ggplot() +
  geom_sf(data = kk_df, aes(fill = total_count_greater_than_1)) +
  labs(title = "Shapefile Plot with Output Data") +
  theme(panel.grid.major = element_blank(),  
        panel.grid.minor = element_blank(),  
        plot.background = element_rect(fill = "white"),  
        panel.background = element_rect(fill = "white"))



setwd("/Users/pratik/Downloads")
# Write otput data to CSV, note to change the year columns from actual csv file , other than that everything is good
write.csv(final_df, file = "Test_data_2011.csv", row.names = FALSE) # set the directory and file name as required