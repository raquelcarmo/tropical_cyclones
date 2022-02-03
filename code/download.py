# !pip install imgaug==0.2.6 cython cartopy geoviews rasterio netcdf4 rioxarray --quiet
# !pip uninstall shapely
# !pip install shapely --no-binary shapely
# !apt-get install libproj-dev proj-data proj-bin  
# !apt-get install libgeos-dev

import os
import bokeh.io
bokeh.io.output_notebook()
import geoviews as gv
import geoviews.feature as gf
gv.extension('bokeh','matplotlib')
import pandas as pd
import numpy as np
import xarray as xr
import rasterio as rio
import rioxarray # geospatial extension for xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm.auto import tqdm
import netCDF4
from glob import glob
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
import cv2
import random
import re


def main(args):
    nc_path = '{}/SAR_swath_nc'.format(args['data_path'])

    # download nc files from Ifremer database
    download(nc_path)

    features = args['features']
    folderName = ''
    for idx, item in zip(range(len(features)), features):
        folderName += '{}_'.format(item) if idx != 2 else item

    # transform nc file into 3-channel image
    nc2image(nc_path, folderName, args)
    del_doubleFiles(os.path.join(args['data_path'], folderName))

    # create csv file containing each image and label
    createCSV(args, os.path.join(args['data_path'], folderName))


def download(nc_path)
    for cat in range(1, 6):
        end_path = '{}/category{}'.format(nc_path, cat)
        os.makedirs(end_path, exist_ok=True)

        # make the request using cyclobs API and store the result in a pandas dataframe
        if cat == 5:
            request_url="https://cyclobs.ifremer.fr/app/api/getData?cat_min=cat-{}&mission=S1B,S1A&product_type=swath&include_cols=all".format(cat)
        else:
            request_url="https://cyclobs.ifremer.fr/app/api/getData?cat_min=cat-{}&cat_max=cat-{}&mission=S1B,S1A&product_type=swath&include_cols=all".format(cat, cat+1)
        df_request = pd.read_csv(request_url)

        # add download path
        df_request['path'] = df_request['data_url'].map(lambda x : os.path.join(end_path, os.path.basename(x)))

        # download 'data_url' to 'path' with wget, and read files
        projection = ccrs.Mercator()
        datasets = []
        for idx, entry in tqdm(df_request.iterrows(), total=df_request.shape[0]):
            ret = os.system('cd %s ; wget -N  %s' % (os.path.dirname(entry['path']), entry['data_url']))

            if ret == 0: 
                ds = xr.open_dataset(entry['path'])
                datasets.append(ds)
                #datasets.append(ds.rio.reproject(projection.proj4_params))
            else:
                # error fetching file
                datasets.append(None)
        #print(datasets)

        df_request['dataset'] = datasets
    
    # gv_list=[gf.coastline.opts(projection=projection)]
    # for ds in df_request['dataset']:
    #     print(ds)
    #     gv_list.append(gv.Image(ds['wind_speed'].squeeze()[::5,::5], crs=projection).opts(cmap='jet',tools=['hover']))
    # gv.Overlay(gv_list).options(width=800, height=500)
    
    # tcInfo = pd.DataFrame(data = {'data_url': df_request['data_url'], 
    #                               'sid': df_request["sid"], 
    #                               'TC_name': df_request["cyclone_name"]})
    # tcInfo.to_csv('{}/TC_info.csv'.format(path), index=False, header=True)


def extractFeature(infoImage, feat, mask=None, mask_dilated=None):
    feature = infoImage.variables[feat][:]

    if mask is not None:
        # mask out the land values
        if mask_dilated is None:
            feature[0][mask[0] != 0] = 0
            feature[0][mask[0] == 1] = 1
        else:
            feature[0][mask_dilated != 0] = 0
    feature = (feature[0] - np.min(feature[0]))/(np.max(feature[0]) - np.min(feature[0]))
    #print("Feature '{}' normalised: max={}, min={}".format(feat, np.max(feature), np.min(feature)))
    #plt.imshow(feature)
    #plt.show()
    return feature


def nc2image(nc_path, folderName, args):
    for cat in range(1, 6):
        # directory where .nc files are
        saved_dir = "{}/category{}".format(nc_path, cat)
        nc_list = glob("{}/*.nc".format(saved_dir))
        count=0

        for nc_product in nc_list:
            # read nc image
            infoImage = netCDF4.Dataset(nc_product, mode='r') 

            try:
                # getting the information of the feature you want
                if args['with_landMask']:
                    # extract mask flag
                    mask = infoImage.variables["mask_flag"][:]
                    #print(np.unique(mask[0]))
                    #plt.imshow(mask[0])
                    #plt.show()

                    if args['dilate_landMask']:
                        mask[mask != 0] = 1
                        kernel = np.ones((11, 11), np.int8)
                        mask_dilated = cv2.dilate(mask[0], kernel, iterations=1)

                # Co-polarization (VV)
                feature_co = extractFeature(infoImage, "nrcs_detrend_co", mask, mask_dilated)

                # Cross-polarization (VH)
                feature_cross = extractFeature(infoImage, "nrcs_detrend_cross", mask, mask_dilated)

                # Wind Speed (WS)
                feature_wind = extractFeature(infoImage, "wind_speed", mask, mask_dilated)

                # Wind Streaks Orientation (WSO) (sinWSO and cosWSO)
                feature_wso = infoImage.variables["wind_streaks_orientation"][:]
                #print(np.nanmax(feature_wso[0]), np.nanmin(feature_wso[0]))
                if np.isnan(feature_wso).any():
                    count +=1
                feature_wso = np.nan_to_num(feature_wso[0])
                #print("feature_wso")
                #plt.imshow(feature_wso[0])
                #plt.show()

                sin_feature_wso = np.sin(feature_wso * np.pi/180.)
                # values from -1 to 1
                sin_feature_wso = (sin_feature_wso - np.min(sin_feature_wso)) / \
                    (np.max(sin_feature_wso) - np.min(sin_feature_wso))

                cos_feature_wso = np.cos(feature_wso * np.pi/180.)
                # values from -1 to 1
                cos_feature_wso = (cos_feature_wso - np.min(cos_feature_wso)) / \
                    (np.max(cos_feature_wso) - np.min(cos_feature_wso))

                # extract longitude and latitude features
                #feature_lon = infoImage.variables["lon"][:]
                #feature_lat = infoImage.variables["lat"][:]

                # extract time registered
                tmax_units = datetime.strptime(infoImage.measurementDate, '%Y-%m-%dT%H:%M:%SZ')\
                                     .strftime('%Y-%m-%d %H_%M_%S')

                # stack matrices along 3rd axis (depth-wise)
                features = args['features']
                feat_dict = {'VV':feature_co, 'VH':feature_cross, 'WS':feature_wind, 
                             'sWSO':sin_feature_wso, 'cWSO':cos_feature_wso}
                output_image = np.dstack((feat_dict[features[0]], feat_dict[features[1]], feat_dict[features[2]]))
                if (output_image < 0).any():
                    output_image = np.clip(output_image, a_min=0, a_max=None)
                #print("First output image:")
                #print('Shape:', output_image.shape)
                #print("Normalised: max={}, min={}".format(np.max(output_image), np.min(output_image)))
                #print("Output without NaNs: max={}, min={}".format(np.nanmax(output_image), np.nanmin(output_image)))
                #plt.imshow(output_image)
                #plt.show()

                infoImage.close()
            except KeyError as err:
                # creating KeyError instance for book keeping
                print("Error:", err)
                infoImage.close()
                continue
            
            # save information extracted
            images_dir = '{}/{}'.format(folderName, category)
            os.makedirs(images_dir, exist_ok=True)
            plt.imsave('{}/{}.png'.format(images_dir, tmax_units), output_image, format='png')
        print('Number of products with NaNs in WSO:', count)


def del_doubleFiles(path):
    for cat in range(5, 1, -1):
        folder_toKeep = '{}/category{}'.format(path, cat)
        files_toKeep = glob(folder_toKeep + '/*.png')

        folder_toDel = '{}/category{}'.format(path, cat-1)
        files_toDel = glob(folder_toDel + '/*.png')

        for f in files_toKeep:
            f = f.split('/')[-1]
            for s in files_toDel:
                s = s.split('/')[-1]
                if f == s:
                    #print('File removed:', s)
                    os.remove(os.path.join(folder_toDel, s))


def createCSV(args, images_dir):
    df = pd.read_csv(args['labels_path'])
    cols = df.columns.tolist()
    for idx, row in df.iterrows():
        # retrieve the coordinates of the bounding boxes
        df['region_shape_attributes'][idx] = re.findall(r'\d+', df['region_shape_attributes'][idx])

    # use a seed so to obtain the same splitting every time
    random.seed(20)

    df_train, df_val, df_test, df_full = [], [], [], []
    for cat in range(1,6):
        dir = '{}/category{}'.format(images_dir, cat)
        lon_dir = '{}/lon'.format(dir)
        lat_dir = '{}/lat'.format(dir)

        image_list = glob('{}/*.png'.format(dir))
        for image_path in image_list:
            #print(image_path)
            image_name = os.path.basename(image_path)

            lon_list = glob('{}/{}.npy'.format(lon_dir, image_name.split('.')[0]))
            lat_list = glob('{}/{}.npy'.format(lat_dir, image_name.split('.')[0]))

            lon = lon_list[0] if lon_list != [] else None
            lat = lat_list[0] if lat_list != [] else None

            aux = [image_path]
            aux.extend([lat, lon])
            
            index = df[df.filename == image_name].index[0]
            bb_present = 1 if df['region_count'][index] == 1 else 0
            aux.append(bb_present)
            aux.append(df['region_shape_attributes'][index])

            df_full.append(aux)
            rand = random.randint(0, 100)
            # split used: 60% for training, 20% for validation, 20% for testing
            if rand < 60:
                df_train.append(aux)
            elif rand < 80:
                df_val.append(aux)
            else:
                df_test.append(aux) 

    col_names = ['image', 'lat', 'lon', 'label', 'bbox_shape']
    df_train = pd.DataFrame(df_train, columns=col_names)
    df_val = pd.DataFrame(df_val, columns=col_names)
    df_test = pd.DataFrame(df_test, columns=col_names)
    df_full = pd.DataFrame(df_full, columns=col_names)

    if (df_full['lat'].values == None).all() and (df_full['lon'].values == None).all():
        #print("Warning: lat and lon variables not considered.")
        df_full.drop(columns=['lat', 'lon'], inplace=True)

    # create path to save the csv files
    csv_path = '{}/csv'.format(images_dir)
    os.makedirs(csv_path, exist_ok=True)

    if df_train.empty == False:
        df_train.to_csv('{}/training.csv'.format(csv_path), index=False)
    if df_val.empty == False:
        df_val.to_csv('{}/val.csv'.format(csv_path), index=False)
    if df_test.empty == False:
        df_test.to_csv('{}/test.csv'.format(csv_path), index=False)
    if df_full.empty == False:
        df_full.to_csv('{}/full_dataset.csv'.format(csv_path), index=False)


if __name__ == "__main__":
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            main(config['data'])
        except yaml.YAMLError as exe:
            print(exe)