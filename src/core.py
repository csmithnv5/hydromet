import os
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import json
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import pathlib as pl
import urllib
import io
from zipfile import ZipFile
from collections import Counter, OrderedDict
import numpy as np
import time
from IPython.display import clear_output
import re
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask

geoDF = 'GeoDataFrame'
plib = 'pathlib.Path'
AEC_METERS = ("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 "
              "+x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs")

def get_huc_12_gdf_from_bigger_huc(huc):
    '''
    takes a HUC12 and then queries USGS HUC-12 services
    to identify all the geometry extent of the
    HUC12 feature.
    Use WWF function if not in the US
    '''
    #intialize lists
    huc12s = []
    #current USGS wbd service
    nhd_service = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?' #base url
    #set parameters for query
    where_c_init = "huc12 LIKE '"+ huc+"%'"
    nhd_param = {'outFields':'*','where':where_c_init,
                 'f':'json','returnGeometry':'true','outSR':4326}
    #get query results
    print('obtaining huc12 basin information')
    local_huc12 = esri_rest_query(nhd_service,nhd_param)
    
    if not local_huc12:
        return ['service error']
    if local_huc12['features']:
        df = json_to_df_huc(local_huc12['features'])
        gdf = df_to_gdf_polygon(df,'geometry',4326)
        return gdf
    
def esri_rest_query(base_url,parameters):
    import json
    r = requests.get(base_url,parameters)
    try:
        response = json.loads(r.content.decode("utf-8"))
        return response
    except:
        print('service error')

def json_to_df_huc(features):
    df = pd.DataFrame(features,columns = features[0].keys())
    df2 = pd.json_normalize(df['attributes'])
    df3 = pd.concat([df.drop(['attributes'], axis=1), df2], axis=1)
    return df3

def df_to_gdf_polygon(df,geo_field,crs):
    df['geometry'] = df['geometry'].apply(lambda x :Polygon(x['rings'][0]))
    gdf = gpd.GeoDataFrame(df).set_geometry(geo_field)
    gdf = gdf.set_crs(epsg=crs)
    return gdf

def get_huc_12_bounds(huc12):
    '''
    takes a HUC12 and then queries USGS HUC-12 services
    to identify all the geometry extent of the
    HUC12 feature.
    Use WWF function if not in the US
    '''
    #intialize lists
    huc12s = []
    #current USGS wbd service
    nhd_service = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?' #base url
    #set parameters for query
    where_c_init = "huc12 = '"+ huc12+"'"
    nhd_param = {'outFields':'*','where':where_c_init,
                 'f':'json','returnExtentOnly':'true','outSR':4326}
    #get query results
    print('obtaining huc12 basin information')
    local_huc12 = esri_rest_query(nhd_service,nhd_param)
    
    if not local_huc12:
        return ['service error']
    if local_huc12['extent']:
        extent_list = list(local_huc12['extent'].values())[:4]
        bounds = ','.join([str(x)[:str(x).find('.')+7] for x in extent_list])
        return bounds
    else:
        print('Not in USA')
        return ['not in USA']
    
def gdf_of_local_precip_gages(bounds,sr,search_distance,desired_stations=['GHCN Daily']):
    rain_gages = []
    base_url = 'https://gis.ncdc.noaa.gov/arcgis/rest/services/cdo/stations/MapServer/'
    parameters = {'geometry':bounds,'geometryType':'esriGeometryEnvelope','f':'pjson','inSR':sr,'outFields':'*','distance':search_distance,'units':'esriSRUnit_Meter','outSR':4269}
    for station in desired_stations:
        level = layer_indexer(base_url,{'f':'pjson'},station)
        results = esri_query(base_url, parameters, level)
        if 'features' in results.keys():
            if results['features']:
                for point in results['features']:
                    point['time_increment'] = station
                    rain_gages.append(point)
    df = pd.DataFrame.from_records(rain_gages)
    if df.empty is False:
        print('Found gages nearby')
        df['STATION_ID'] = df['attributes'].apply(lambda x: x['STATION_ID'])
        df['STATION_NAME'] = df['attributes'].apply(lambda x: x['STATION_NAME'])
        df['DATA_BEGIN_DATE'] = df['attributes'].apply(lambda x: pd.to_datetime(x['DATA_BEGIN_DATE'],unit='ms'))
        df['DATA_END_DATE'] = df['attributes'].apply(lambda x: pd.to_datetime(x['DATA_END_DATE'],unit='ms'))
        df['lat'] = df['attributes'].apply(lambda x: x['LATITUDE'])
        df['long'] = df['attributes'].apply(lambda x: x['LONGITUDE'])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
        gdf.set_crs(epsg=4269,inplace=True)
        return gdf
    else:
        print(f'No gages nearby at {search_distance} meter search distance')
        return df
    
def layer_indexer(url: str, params: dict, layer: str) -> int:
    '''
    Searches through base url and returns the layer corresponding to the map type selected (effective, preliminary, pending)

    Parameters:
    -----------
    url: str
        base url to find fema hazard map data
    params: dict
        parameters to pass in api request
    layer: str
        layer for which id will be extracted

    Returns:
    --------
    layer id: int
        layer number for selected fema map type

    '''
    r = requests.get(url, params)
    rest_layers = json.loads(r.content)['layers']
    for l in rest_layers:
        if l['name'] == layer:
            return l['id']
        
def esri_query(base_url, parameters: dict, level: int) -> dict:
    '''
    Use the desired MapServer to get a REST response.

    Parameters:
    -----------
      base_url: string of base arcgis REST server
      patermeters: dict
         dictionary containing the parameters to be passed to the query. This dictionary is passed by the `get_huc12` function.
      level: int
         indicating the layer level to be returned. 

    Return:
    --------
      response: dict
        a JSON decoded object that can be accessed like a dictionary.
    '''
    query_url = base_url+str(level)+'/query?'
    r = requests.get(query_url, parameters)
    try:
        response = json.loads(r.content)
        return(response)
    except Exception as error:
        print(f'Except raised trying to run ESRI query with params {parameters} : {error}') 

def add_to_gage(df):
    df['Period of Record'] = df[['DATA_END_DATE','DATA_BEGIN_DATE']].apply(lambda x: (pd.to_datetime(x[0])-pd.to_datetime(x[1])).days/365.25,axis=1)
    df['link_works'] = df['STATION_ID'].apply(lambda x: "yes" if requests.get(f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{x[x.find(':')+1:]}.csv").status_code == 200 else "no")
    return df

def add_to_gage_hourly(df):
    df['Period of Record'] = df[['DATA_END_DATE','DATA_BEGIN_DATE']].apply(lambda x: (pd.to_datetime(x[0])-pd.to_datetime(x[1])).days/365.25,axis=1)
    df['link_works'] = df['STATION_ID'].apply(lambda x: "yes" if requests.get(f"https://www.ncei.noaa.gov/data/coop-hourly-precipitation/v2/access/USC00{x[x.find(':')+1:]}.csv").status_code == 200 else "no")
    return df

def plot_rain_gages(area_gdf,gage_gdf):
    '''
    Docstring for plot_rain_gages
    
    :param area_gdf: Description
    :param gage_gdf: Description
    '''
    col = None if 'huc12' not in area_gdf.columns.to_list() else 'huc12'
    fig, ax = plt.subplots(figsize=(15, 15))
    if col:
        area_gdf.to_crs(3857).plot(ax=ax, alpha=0.7, cmap="plasma",cax=col,column=col,legend=True)
    else:
        area_gdf.to_crs(3857).plot(ax=ax, alpha=0.7,legend=True)
    gage_gdf.to_crs(3857).plot(ax=ax)
    for x, y, label in zip(gage_gdf.to_crs(3857).geometry.x, gage_gdf.to_crs(3857).geometry.y, gage_gdf.STATION_ID):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
        ax.add_artist(ScaleBar(1,location="lower left"))

def download_file(url,out_dir):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_dir/local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return out_dir/local_filename

    
def check_attributes(gdf: geoDF) -> None:
    '''Checks the passed geodataframe to make sure that "Volume" and "Region"
       are not attributes, else the intersect_temporal_areas function will 
       fail. 
    '''
    assert 'Volume' and 'Region' not in list(gdf.columns), ('"Volume" and '
        '"Region" cannot be columns in the vector polygon. Rename columns '
        'and reload')

def plot_area_of_interest(geo_df: geoDF, select_data: str, 
                                                ) -> plt.subplots:
    '''Plots the column of the geodataframe with matplotlib.
    '''
    fig = geo_df.plot(categorical = True, figsize = (10, 14))
    fig.set_title('Area of Interest (ID: {})'.format(select_data))
    fig.grid()


def intersect_temporal_areas(geo_df: geoDF, datarepository_dir: plib, 
           Temporal_area_filename: str, alldata: bool=False, 
           projected_crs: str = AEC_METERS) -> (dict, geoDF):
    '''Intersects the area of interest with the NOAA Atlas 14 volumes and 
       regions. The volume, region, and percent area of the area of interest 
       is returned in a dictionary. If alldata is set to True, the dictionary
       returned will contain information for all volumes and regions that 
       interesect the area of interest. The keys 'Volume', 'Region', and 
       'Percent_area' will always represent the NOAA Atlas 14 volume and 
       region that has the largest intersection with the area of interest.
    '''
    vol_gdf = gpd.read_file(datarepository_dir/Temporal_area_filename)
    vol_gdf.to_crs(AEC_METERS, inplace = True)
    geo_df_projected = geo_df.to_crs(AEC_METERS)
    intersection = gpd.overlay(geo_df_projected, vol_gdf, how='intersection')
    intersection['area'] = intersection['geometry'].apply(lambda x: x.area)
    t_area = sum(intersection['area'])
    intersection['p_area'] = intersection['area'].apply(lambda x: x/t_area*100)
    intersection = intersection.sort_values('p_area', ascending=False)
    intersection = intersection.reset_index(drop=True)
    d = {}
    for i in intersection.index:
        if i == 0:
            d['Volume'] = intersection.loc[i, 'Volume']
            d['Region'] = intersection.loc[i, 'Region']
            d['Percent_area'] = intersection.loc[i, 'p_area']
        elif i>0 and alldata:
            d[f'Volume_{i}'] = intersection.loc[i, 'Volume']
            d[f'Region_{i}'] = intersection.loc[i, 'Region']
            d[f'Percent_area_{i}'] = intersection.loc[i, 'p_area']
    for k,v in d.items():
        print('{:<17s}{:>1s}'.format(str(k),str(v)))
    intersection.to_crs(geo_df.crs, inplace = True)    
    return OrderedDict(d), intersection


def get_volume_code(datarepository_dir: str, vol_code_filename: str, 
                                        vol: int, sub_vol: int = None) -> str:
    ''' Extracts the NOAA Atlas 14 volume code for the specified volume number.
    '''
    if vol==5: assert sub_vol!=None, 'For Volume 5, specify sub-volume number'
    orig_dir = os.getcwd()
    os.chdir(datarepository_dir)   
    with open(vol_code_filename) as json_file:  
        vol_code = json.load(json_file)
    code = vol_code[str(vol)]
    if vol == 5: code = code[str(sub_vol)]
    os.chdir(orig_dir)
    print('NOAA Atlas 14 Volume Code:', code)
    return code


def build_precip_table(geo_df: geoDF, all_zips_list: list, noaa_url: str, 
    vol_code: str, num_attempts: int=10, verbose: bool=True) -> pd.DataFrame:
    '''Calculates the area-averaged precipitation for each return frequency 
       and duration contained within the list of zipfiles.
    '''
    start = time.time()
    results = []
    for i, zip_name in enumerate(all_zips_list):
        clear_output(wait=True)
        remote_file = os.path.join(noaa_url, zip_name)
        get_remote_file = True
        count = 1
        while get_remote_file and count<=num_attempts:
            try:
                open_socket = urllib.request.urlopen(remote_file)
                get_remote_file = False
            except:
                if verbose: print("Unable to get data on attempt {1} for "
                                                "{0}".format(zip_name, count))
                count+=1
        memfile = io.BytesIO(open_socket.read())
        with ZipFile(memfile, 'r') as openzip:
            gridfiles = openzip.namelist()
            mes = "Expected to find 1 file, found {0}".format(len(gridfiles))
            assert len(gridfiles) == 3, mes
            local_file = gridfiles[0]
            f = openzip.open(local_file)
            content = f.read() 
            local_file_disk = os.path.join(os.getcwd(), local_file)
            with open(local_file_disk, 'wb') as asc:
                asc.write(content)
        grid_data = parse_filename(zip_name, vol_code)
        try:
            grid_data['value'] = get_masked_mean_atlas14(geo_df, local_file_disk)   
            results.append(grid_data)
            os.remove(local_file_disk)
            if verbose: 
                
                print(i, zip_name)
        except ValueError as e:
            if 'Input shapes do not overlap' in str(e):
                print(f'{e} - if two volumes were identified, try using the other volume')
                os.remove(local_file_disk)
                raise e
    df = pd.DataFrame.from_dict(results)
    assert df.isnull().values.any()!=True, 'NaN in results dataframe'
    if verbose: 
        print(i)
        print(round(time.time()-start), 'Seconds')
        print(display(df.head()))
    return df


def parse_filename(zip_name: str, reg: str) -> dict:
    '''Builds a dictionary with the region, recurrance interval, duration, 
       and statistic type using the zip_name and region.
    '''
    dic = {'a': 'Expected Value', 'al': 'Lower (90%)', 'au': 'Upper (90%)'}
    reg = zip_name[0:re.search(r"\d", zip_name).start()]
    TR = zip_name.split(reg)[1].split('yr')[0]
    dur = zip_name.split('yr')[1].split('a')[0]
    stat = zip_name.split(dur)[1].replace('.zip','')
    grid_data = {'region':reg, 'TR':TR, 'duration':dur, 'statistic':dic[stat]}
    return grid_data    


def get_masked_mean_atlas14(gdf: geoDF, raster: str) -> float:
    '''Masks the Atlas 14 precipitation raster by the passed polygon and then 
       calculates the average precipitation for the masked polygon.
    '''
    geoms = gdf.geometry.values
    geoms = [mapping(geoms[0])]
    with rasterio.open(raster) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        raw_data = out_image[0]
        region_mean = raw_data[raw_data != src.nodatavals ].mean()
    mean_m = region_mean*0.001    
    return mean_m

def plot_aoi_noaa_intersection(intersection_gdf: geoDF, 
                                            select_data: str) -> plt.subplots:
    '''Plots the intersection of the geodataframe and the NOAA Atlas 14 
       volumes and regions.
    '''
    intersection_gdf['Volume_Region'] = 'Volume: ' +\
                        intersection_gdf['Volume'].map(str) + ', Region: ' +\
                                        intersection_gdf['Region'].map(str)
    fig = intersection_gdf.plot(column='Volume_Region', categorical=True, 
                                                figsize=(10, 14), legend=True)
    fig.set_title('Area of Interest (ID: {}) by NOAA Atlas' 
                                                'Region'.format(select_data))
    fig.grid()

def get_noaa_precip_values(vol_code,durations, verbose = True):
    noaa_url = 'https://hdsc.nws.noaa.gov/pub/hdsc/data/{}/'.format(vol_code)

    req = urllib.request.Request(noaa_url)
    data = urllib.request.urlopen(req).read().decode().split()

    copy_zips = []
    for duration in durations:
        zips = [d for d in data if ('.zip' in d) and ('{}'.format(duration) in d) and ('ams' not in d)]
        copy_zips.append(zips)
        if verbose: 
            print('{} files found for {}'.format(len(zips), duration))

    all_zips_list = list(np.array(copy_zips).flat)

    for i, zip_name in enumerate(all_zips_list):
        all_zips_list[i]= zip_name.split("\"", 1)[1].split("\"", 1)[0]
    return all_zips_list, noaa_url

def events_initialize(events_lib:str):
    
    if events_lib == 'FEMA':
        fema_intervals = ['10', '25', '50', '100', '500','100_minus','100_plus']  # Return intervals for FEMA study
        #recurrence_intervals = np.array([2, 5, 10, 25, 50, 100, 200, 500, 1000])  # Return intervals for calculating runoff values.
        hydro_events_dict = {'10':10, '25':25, '50': 50, '100':100, '500':500,'100_minus': 100, '100_plus': 100}
        return fema_intervals, hydro_events_dict
    if events_lib == 'NOAA':
        recurrence_intervals = np.array([2, 5, 10, 25, 50, 100, 200, 500, 1000])  # Return intervals for calculating runoff values.
        return recurrence_intervals
    
def nrcs_nesting_eqn(t:int,cumulative_ratio:pd.DataFrame,dur_precip_df_noaa:pd.DataFrame):
    '''
    From NRCS-Part-630-NEH-Chapter-4-rainfall.pdf
    The design rainfall 
    distribution is developed to have the 100-year 24-hour rainfall, the 100-year 12-hour rainfall, 
    etc., down to the 100-year 5-minute rainfall imbedded in a single storm
    '''
    # Step 3 
    #multiplied 100th to t threshold to account for floats
    if t <= 90:
        #eqn 4-1
        a = (2/3*cumulative_ratio['9'] - cumulative_ratio['6'])/18
        b = (cumulative_ratio['6'] - 36*a)/6
        crr = a*((t/10)**2) + b*(t/10)
        
    elif t<=105:
        #eqn 4-2
        a2 =  (9/10.5 * cumulative_ratio['10.5'] - cumulative_ratio['9']) / 13.5
        b2 = (cumulative_ratio['9'] -81*a2) / 9
        crr = a2*((t/10)**2)+b2*(t/10)

    elif t<= 115:
        #eqn 4-3
        a3 = 2 * (cumulative_ratio['11.5'] - 2*cumulative_ratio['11'] + cumulative_ratio['10.5'])
        b3 = cumulative_ratio['11.5'] - cumulative_ratio['10.5'] - 22*a3
        c3 = cumulative_ratio['11'] - 121*a3 - 11*b3
        crr = a3 * ((t/10)**2) +b3*(t/10)+c3

    elif t<= 117:
        #test with 11.6
        intensity_115 = (cumulative_ratio['11.5'] - cumulative_ratio[114])/0.1
        if t <= 116:
            factor_t = -0.867*intensity_115 +0.4337       
        elif t <= 117:
            factor_t = -0.4917*(intensity_115) +0.8182
            if factor_t > 0.799:
                factor_t = 0.799
        crr = cumulative_ratio['11.5']  + factor_t*(cumulative_ratio['11.75']-cumulative_ratio['11.5'])
    elif t<= 118:
        #CRR(11.8)=CRR(11.75)+(11.8–11.75)/(11.875–11.75)(CRR(11.875)–CRR(11.75))
        crr = cumulative_ratio['11.75']+ 0.4*(cumulative_ratio['11.875']-cumulative_ratio['11.75'])
    elif t<= 119:
        #CRR(11.9)=CRR(11.875)+(11.9–11.875)/(11.9167–11.875)(CRR(11.9167)–CRR(11.875))
        crr = cumulative_ratio['11.875']+0.6*(cumulative_ratio['11.9167']-cumulative_ratio['11.875'])                                     
    elif t <= 120:
        #6-min / 24-hr ratio = 5-min / 24-hr ratio + 0.2 (10-min /24-hr ratio – 5-min /24-hr ratio)
        #Ratio(12.0) = Ratio(12.1) - 6-minute/24-hour ratio
        six_min_ratio = dur_precip_df_noaa.loc['05m'].ratio +0.2*(dur_precip_df_noaa.loc['10m'].ratio-dur_precip_df_noaa.loc['05m'].ratio)
        t_ahead = 121
        crr_ahead = 1 - cumulative_ratio[(240 - t_ahead)]
        crr = crr_ahead - six_min_ratio
    else:
        assert t>120, 'something is wrong with the if elif statements above'
        crr = 1 - cumulative_ratio[(240 - t)]
    #save to dictionary for second half look up
    cumulative_ratio[t] = crr

    return crr

