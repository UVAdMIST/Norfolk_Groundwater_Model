"""
This script downloads data for a single variable with a specified date and time from the Utah HRRR archive using cURL.
Using 'APCP' as the variable gets the hourly precipitation amount for each forecast hour.
Utah Archive web page: http://home.chpc.utah.edu/~u0553130/Brian_Blaylock/hrrr_FAQ.html

Steps:
    1) Read the lines from the Metadata .idx file
    2) Identify the byte range for the variable of interest
    3) Download the byte range using cURL.
"""

import re
from datetime import date
import os
import sys
import shutil
import struct
import urllib2, ssl
from osgeo import gdal, ogr
from os.path import basename
from os.path import splitext
import numpy as np

# =============================================================================
#     Modify these
# =============================================================================
DATE = date(2016, 9, 6)    # Model run date YYYY,MM,DD
hour = 20                # Model initialization hour
fxx = range(19)            # Forecast hour range
                           # Note: Valid Time is the Date and Hour plus fxx.

model = 'hrrr'        # ['hrrr', 'hrrrX', 'hrrrAK']
field = 'sfc'              # ['sfc', 'prs']

var_to_match = 'APCP'      # must be part of a line in the .idx file
                           # Check this URL for a sample of variable names you can match:
                           # https://api.mesowest.utah.edu/archive/HRRR//oper/sfc/20170725/hrrr.t01z.wrfsfcf00.grib2.idx
# =============================================================================
# =============================================================================

# create directories to store data
direct = 'C:/HRRR/HRRR_Archive_'+DATE.strftime('%Y%m%d')+'_hr_'+str(hour)
os.makedirs(direct)
os.makedirs(direct+"/GRIB2")
os.makedirs(direct+"/TIF")

rainfall_data = np.zeros((19,8))
shp_filename = 'C:/Users/Ben Bowes/Documents/HRSD GIS/Shallow Wells/Norfolk_Wells_UTM.shp'

for i in fxx:
    rainfall_data [i,0] = i
    # Rename the file based on the info from above (e.g. 20170310_h00_f00_TMP_2_m_above_ground.grib2)
    outfile_grib = direct + '/GRIB2/%s_h%02d_f%02d_%s.grib2' % (DATE.strftime('%Y%m%d'), hour, i,
                                                                var_to_match.replace(':', '_').replace(' ', '_'))
    outfile_tif = direct + '/TIF/%s_h%02d_f%02d_%s.tif' % (DATE.strftime('%Y%m%d'), hour, i,
                                                           var_to_match.replace(':', '_').replace(' ', '_'))
    outfile_tif_prj = direct + '/TIF/%s_h%02d_f%02d_%s_projected.tif' % (DATE.strftime('%Y%m%d'), hour, i,
                                                                         var_to_match.replace(':', '_').replace(' ', '_'))

    # Model file names are different than model directory names.
    if model == 'hrrr':
        model_dir = 'oper'
    elif model == 'hrrrX':
        model_dir = 'exp'
    elif model == 'hrrrAK':
        model_dir = 'alaska'

    # This is the URL to download the full GRIB2 file. We will use the cURL command
    # to download the variable of interest from the byte range in step 3.
    pandofile = 'https://pando-rgw01.chpc.utah.edu/%s/%s/%s/%s.t%02dz.wrf%sf%02d.grib2' \
                % (model, field, DATE.strftime('%Y%m%d'), model, hour, field, i)

    # This is the URL with the Grib2 file metadata. The metadata contains the byte
    # range for each variable. We will identify the byte range in step 2.
    sfile = pandofile+'.idx'

    # 1) Open the Metadata URL and read the lines
    request = urllib2.Request(sfile)
    response = urllib2.urlopen(request, context = ssl._create_unverified_context())
    # idxpage = urllib2.urlopen(sfile) #certificates are not validating correctly
    lines = response.readlines()

    # 2) Find the byte range for the variable. First find where the
    #    variable is located. Keep a count (gcnt) so we can get the end
    #    byte range from the next line.
    gcnt = 0
    for g in lines:
        expr = re.compile(var_to_match)
        if expr.search(g):
            parts = g.split(':')
            rangestart = parts[1]
            parts = lines[gcnt+1].split(':')
            rangeend = int(parts[1])-1
            print 'range:', rangestart, rangeend
            byte_range = str(rangestart) + '-' + str(rangeend)

            # 3) When the byte range is discovered, use cURL to download.
            os.system('curl -s -o %s --range %s %s' % (outfile_grib, byte_range, pandofile))
            os.system('curl -s -o %s --range %s %s' % (outfile_tif, byte_range, pandofile))
            print 'downloaded', outfile_tif

        gcnt += 1


# project hrrr data to same projection as wells and extract data for those points only
    os.system('gdalwarp %s %s -t_srs "+proj=utm +zone=18 +datum=NAD83" -tr 500 500' % (outfile_tif, outfile_tif_prj))
    # Open the TIF file
    src_ds = gdal.Open(outfile_tif_prj)
    if src_ds is None:
        print "Failed to open the raster.\n"
        sys.exit(1)
    gt = src_ds.GetGeoTransform()
    rb = src_ds.GetRasterBand(1)

    # Open the shapefile
    ds = ogr.Open(shp_filename, 1)
    if ds is None:
        print "Failed to open the study area well shapefile.\n"
        sys.exit(1)
    lyr = ds.GetLayerByName(splitext(basename(shp_filename))[0])
    if lyr is None:
        print "Error opening the shapefile layer"
        sys.exit(1)

    # Check that all 7 wells exist
    if lyr.GetFeatureCount() != 7:
        raise Exception(lyr.GetFeatureCount(), 'Not equal to number of wells'
                                               ' in the study area, 7 wells')

    # Extract the forecast rainfall data from the TIF file underneath each well
    j = 1
    for feat in lyr:
        geom = feat.GetGeometryRef()
        mx, my = geom.GetX(), geom.GetY()
        # Convert from map to pixel coordinates. That takes less memory size and less computation time.
        # Only works for geo-transforms with no rotation.
        # x pixel
        px = int((mx - gt[0]) / gt[1])
        # y pixel
        py = int((my - gt[3]) / gt[5])

        # Using the Struct library to unpack the return value for each pixel
        structval = rb.ReadRaster(px, py, 1, 1, buf_type=gdal.GDT_Float64)
        pix_val = struct.unpack('d', structval)
        rain_val = float(pix_val[0])
        well_name = feat.GetField(feat.GetFieldIndex('MONITORING'))
        print well_name, rain_val
        rainfall_data[i,j] = rain_val*0.0393701
        j += 1


print rainfall_data

np.savetxt(direct + '/rainfall.csv', rainfall_data, delimiter=',',
           header='Forecast,MMPS-175,MMPS-129,MMPS-155,MMPS-043,MMPS-125,MMPS-170,MMPS-153')


# once rainfall is extracted, delete large grib and tif folders
try:
    shutil.rmtree(direct+"/GRIB2")
    shutil.rmtree(direct+"/TIF")
except OSError, e:
    print ("Error: %s - %s." % (e.filename, e.strerror))