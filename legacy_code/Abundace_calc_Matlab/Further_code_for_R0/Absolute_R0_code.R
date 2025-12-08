
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
## first we will convert daily data to monthly data 


# To check the that how monthly avaerage looks like (is it correct or not), whether lat long combinations are good with monthly data

#r0_one1 = rasterFromXYZ(data.frame(unique_combinations,result_matrix=result_matrix1[,6]))    # always use unique combination here 
#plot(r0_one1)

## so now result matrix contains vectorial capacity data ans abundance contain the result matrix 1
## Now combining them with other parameters to get absolute R0

## first extracting the grid area  for the region we are concern from total grid area file

# extracting the lat long of the file for which we want to extract, only do this if you are dealing with 0.5*0.5 resolution otherwise
# extract and calculate area of grid from the program I have written (in brazil code analysis), we can also compute it from there and directly use it


## read in  grid area for scaling the abundance
g_area<-nc_open("/Users/pratik/Desktop/R0_computations/Input_files/R0_files/gridarea.nc")

area_grid<-ncvar_get(g_area,"cell_area")

lat_ga<-ncvar_get(g_area,"lat")
lon_ga<-ncvar_get(g_area,"lon")

colnames(area_grid)<-lat_ga
row.names(area_grid)<-lon_ga

area_grid_long<-melt(area_grid)
names(area_grid_long)<-c("x","y","area")

ras_area<-rasterFromXYZ(area_grid_long)

proj4string(ras_area)<-CRS("EPSG:4326")

plot(ras_area)




lat<-ncvar_get(tmax,"latitude")
lon<-ncvar_get(tmax,"longitude")

ras_coords<-expand.grid(x=lon,y=lat)    # for unequal number of rows and columns 

ras_coords1<-ras_coords

coordinates(ras_coords1)<-~x+y

proj4string(ras_coords1)<-CRS("EPSG:4326")

area_coord<-raster::extract(ras_area,ras_coords1)

ras_coord_Area<-data.frame(ras_coords,area_m2=area_coord)

raster_coord_Area<-rasterFromXYZ(ras_coord_Area)

raster_coord_Area$area_km2<-raster_coord_Area$area_m2/1e6

plot(raster_coord_Area[[2]])

############
## read in  grid area for scaling the abundance, if we have already area nc file from code I have written in Brazil code analysis
#g_area<-nc_open("grid_area_15_min_0.25_reso_Europe.nc")

#area_grid<-ncvar_get(g_area,"gridarea")

#area_grid = area_grid[,,1]      # only select the first layer to keep it as 2d matrix


#colnames(area_grid)<-lat_ga
#row.names(area_grid)<-lon_ga

#area_grid_long<-melt(area_grid)   # create a datframe of lat long combination

#names(area_grid_long)<-c("x","y","area_m2")   #changing the name of columns , to avoid confusion
################

area_grid_long = ras_coord_Area


month_yr<-expand.grid(month=1:12,year= 2022) %>%   # for only one year from 1st month to 12th month
  mutate(n=1:n())

#month_yr <- expand.grid(month = 1:12, year = 2015:2016) %>%      # just to expand all the months and years of our combination, all the months of 2015 and only first month of 2016
#  mutate(n = 1:n()) %>%
#  filter(year == 2015 | (year == 2016 & month == 1))



## now replicating this are according to months of abundance or vectorial capacity
area_m2_2500 = replicate(nrow(month_yr),area_grid_long$area_m2)/2500  # so now area/2500 will be replicated over the 12 months of year 2015 and one month of year 2016
  
area_km2<- replicate(nrow(month_yr),area_grid_long$area_m2)/1e6    # so now area in km be replicated over the 12 months of year 2015 and one month of year 2016

# common_Files_generator()
##create different pop and pop density and scale factor

pop_nc<-"pop_count_2020_EU_0.25.nc"  # population data path

pop_oc<-nc_open(pop_nc)

pop_month1<-ncvar_get(pop_oc,"pop")

#replacement_value <- 1e+38

#pop_month[is.na(pop_month)] <- replacement_value   # assigning very large values to NA values 

#pop_month1 = pop_month1[,,1]    # Only the first slice of matrix

# now make it replicate in the same way as area matrix for total of 13 months


colnames(pop_month1)<-lat
row.names(pop_month1)<-lon

pop_grid_long<-melt(pop_month1)   # create a datframe of lat long combination

names(pop_grid_long)<-c("x","y","pop")   #changing the name of columns , to avoid confusion

pop_month = replicate(nrow(month_yr),pop_grid_long$pop)  # so now population will be replicated over the 12 months of year 2015 and one month of year 2016


#for now use 2.9 and 1000
#par.a<-2.9
#pAr.pop_breed_per_person<-1000


pAr.pop_breed_per_person<-1000
area_by_2500<-area_m2_2500/pAr.pop_breed_per_person     # now this is complete c, what we are interested in 
dim(area_by_2500)
dim(pop_month)
scale_factor1<-(pop_month[,]^2)/(area_by_2500[,]^2+pop_month[,]^2)  # computing the g(p,c) function
scale_factor<-ifelse(is.na(scale_factor1)|scale_factor1<0,0,scale_factor1)
dim(scale_factor)


## now further combining everything

   #vc_op<-nc_open(vc_file_run$path[p])
  #vc_fl<-as.matrix(ncvar_get(vc_op,'vc'))[,-c(1:12)]
  #vc_op <- nc_open(vc_file_run$path[1])
  #vc_data <- ncvar_get(vc_albo, 'vc')
  #vc_data <- vc_data[, -(1:12)]
  vc_fl <- as.matrix(final_matrix_vc)
  dim(vc_fl)                 
  #gc()
  #Abund_path<-"/Users/pratik/Desktop/R0_computations/Abundance/big_memory_files/"
  #abund_files<-list.files(Abund_path,full.names =T,pattern =".desc")
  
  #if(str_detect(tolower(vc_file_run$fname[p]),"aegypti|aegpyti|zika")){
   # par.a<-2.9
   # idx<-grep("aegypti|aegpyti",tolower(abund_files))
  #}else{
   # par.a<-2.9
   # idx<-grep("albopictus",tolower(abund_files))
  #}
  
  #abund_Mat_<-t(as.matrix(attach.big.matrix(abund_files[idx])))[,-(1:7)]
  #dim(abund_Mat_)
  abund_Mat<-ifelse(is.na(final_matrix_abun)|final_matrix_abun<0,0,final_matrix_abun)
  #rm(abund_Mat_)
  #gc()
  par.a<-2.9        # old estimate of constant a from Sewe estimates
  w<-par.a* 20      # UPDATED VALUE OF THE CONSTANT FROM REAL DENGUE OUTBREAKS.
  mean_M<-w/as.numeric(apply(abund_Mat,2,FUN=function(x) mean(x[x>0])))   #column wise mean of whole long lat combination, In Sewe's code mean of only x>0 colums were taken, I don't know why
   #mean_M = 4
  dim(abund_Mat)
  length(mean_M)
  #mean_M_A<-t(replicate(67420,mean_M))
  
  #mean_M_A<-t(replicate(67420,mean_M))
  #rm(mean_M_A) gc()
  #table(t(t(abund_Mat)*mean_M)==abund_Mat*mean_M_A)
  #m_ratio<-t(t(abund_Mat)*mean_M)*scale_factor
  #R0<-vc_fl*m_ratio rm(m_ratio) gc()
  R0<-vc_fl*t(t(abund_Mat)*(mean_M/1000))*scale_factor
  
  ## save R0

  R0 = ifelse(is.na(R0)|R0<0,0,R0)
  rast_R0 = rasterFromXYZ(data.frame(ras_coords,R0=R0))
  proj4string(rast_R0)<-CRS("EPSG:4326")
  plot(rast_R0)

  
  #Plottand and extracting the data
  
  ## shapefile only for good plotting
  
  setwd("/Users/pratik/Desktop/ae_albopictus_model")
  
  load("map_selected.Rdata")
  ls()
  
  geometry = map_selected$geometry
  #shapefile1 <- st_read(sf_map)
  
  sf_map <- st_sf(map_selected, crs = 4326)
  
  shapefile <- st_transform(sf_map, crs = "+proj=longlat +datum=WGS84")
  
  
  
  #setwd("/Users/pratik/Desktop/ae_albopictus_model")
  # tempo = rast("Mosquito_abundance_EU_22010.nc")
  #tempo = rast ("era5land_tmean_EU_daily_0.25_2010.nc")
  tempo = rast_R0[[8]]
  
  ### assigning crs to raster data
  #proj4string(tempo)<-CRS("EPSG:4326")
  
  r0_one <- tempo
  
  #Cropping the data so that it matches with area covered by shapefile i.e. both have same lat long extension
  r0_EU <- crop(r0_one, extent(shapefile))
  #we can print r0_usa to check extent of latitude and longitude
  
  #to assign same CRS to ro_usa as that of shapefile if they don't have same CRS
  crs(r0_EU) <- crs(shapefile)
  
  #if we want to know all the values of r0_usa with a region of shapefile we can use below command it will give all the
  #r0_usa for a single geometry of shapefile. Note that ID number 1 corresponds to first row of shapefile, and if say ID
  # number 6 has multiple values then all the values fall to 6th geometry of shapefile. The total number of ID's will
  # be eqaul to rows of shapefi;e
  #r0_values <- extract(r0_usa, shapefile, df = TRUE)
  
  # to get R0 values based on weighatge area it covered within region this has rows equal to shapefile row 
  # remeber to import library(exactextractr)
  output_data = exact_extract(r0_EU,shapefile,
                              fun="weighted_mean",weights="area")
  
  #output_data <- exact_extract(r0_EU, shapefile)
  
  
  #values_at_points <- extract(r0_EU,shapefile)
  values_at_points = as.numeric(output_data)
  
  shapefile$values <- values_at_points
  
  
  
  
  # Define your custom color palette with three colors
  custom_colors <- c("navyblue", "white", "red4")
  
  # Set your upper limit
  upper_limit <- 1
  
  
  # Set legend title
  legend_title <- "Abundance"
  
  # for no space after highest value in legend
  
  # Plot the shapefile with output_data using a custom color scale
  ggplot() +
    geom_sf(data = shapefile, aes(fill = ifelse(values > upper_limit, upper_limit, values))) +
    labs(title = "Temperature on 14th_July_2022 ") +
    scale_fill_gradientn(colors = custom_colors, limits = c(0, upper_limit), name = legend_title) +
    theme_minimal() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank()
    )
  
  
  
  # shapefile good for extracting the data of complete Europe according to LCDE context
  
  #reading the shapefile
  setwd("/Users/pratik/Downloads")
  shp_file <- "NUTS_RG_01M_2021_3035_LEVL_3.shp"
  
  # Read the shapefile
  shapefile1 <- st_read(shp_file)
  
  shapefile <- st_transform(shapefile1, crs = "+proj=longlat +datum=WGS84")
  
  ### assigning crs to raster data
  proj4string(rast_R0)<-CRS("EPSG:4326")
  
  r0_one <- rast_R0
  
  #Cropping the data so that it matches with area covered by shapefile i.e. both have same lat long extension
  r0_usa_EU <- crop(r0_one, extent(shapefile))
  #we can print r0_usa to check extent of latitude and longitude
  
  #to assign same CRS to ro_usa as that of shapefile if they don't have same CRS
  crs(r0_usa_EU) <- crs(shapefile)
  
  #if we want to know all the values of r0_usa with a region of shapefile we can use below command it will give all the
  #r0_usa for a single geometry of shapefile. Note that ID number 1 corresponds to first row of shapefile, and if say ID
  # number 6 has multiple values then all the values fall to 6th geometry of shapefile. The total number of ID's will
  # be eqaul to rows of shapefi;e
  #r0_values <- extract(r0_usa, shapefile, df = TRUE)
  
  # to get R0 values based on weighatge area it covered within region this has rows equal to shapefile row 
  # remeber to import library(exactextractr)
  output_data = exact_extract(r0_usa_EU,shapefile,
                              fun="weighted_mean",weights="area")
  
  # Write otput data to CSV, if you want to see the data
  #write.csv(output_data, file = "output_values.csv", row.names = FALSE)
  
  # Combine shapefile details in a data frame
  shp_to_df <- as.data.frame(shapefile)
  req_df <- shp_to_df[,1:7]    # only required information from shapefile not geometry
  
  
  # Add the output data to the 'req_df' dataframe
  #final_df <- cbind(req_df, output_data)
  
  num_rows <- nrow(req_df)     # extracting the number of rows of of shapefile or number of NUTS3 reason in Europe 
  
  # Initialize two empty matrix to store aggregated values (final values) yearly of R0 and LTS with rows equal to number of NUTS 3 reason
  final_matrix_LTS <- matrix(nrow = num_rows, ncol = 0)
  final_matrix_R0 <- matrix(nrow = num_rows, ncol = 0)
  
  years = (ncol(output_data))/12
  
  for (i in 1:years) {
    
    # Compute total count of values greater than 1 in each row
    LTS_year <- apply(output_data[, (12*(i-1)+1):(12+12*(i-1))], 1, function(row) sum(row > 1))
    
    # Add the result to final_LTS matrix
    final_matrix_LTS <- cbind(final_matrix_LTS, as.matrix(LTS_year))   
    
    # Compute total aggregated R0 in each row for each year
    R0_year <- rowMeans(output_data[, (12*(i-1)+1):(12+12*(i-1))])
    
    # Add a new column with the total count of values greater than 1 in each row
    final_matrix_R0 <- cbind(final_matrix_R0, as.matrix(R0_year))
      
    # Print the updated data frame
    #print(final_df[,20])
  }
  
  
  
  # Now add the LTS and R0 data to the 'req_df' dataframe of shapefile to have proper format
  
  final_R0 <- cbind(req_df, final_matrix_R0)
  final_LTS <- cbind(req_df, final_matrix_LTS)
  
  
  ## now save the data as csv file
  
  
  setwd("/Users/pratik/Downloads")
  # Write otput data to CSV, note to change the year columns from actual csv file , other than that everything is good
  write.csv(final_R0, file = "final_R0.csv", row.names = FALSE) # set the directory and file name as required
  write.csv(final_LTS, file = "final_LTS.csv", row.names = FALSE) # set the directory and file name as required
 
 # !!! CHANGE THE COLUMNNAMES OF ABOVE FILES ACCORDING TO YEAR USED IN COMPUTATIONS
  
  
  
  
  
  
  
