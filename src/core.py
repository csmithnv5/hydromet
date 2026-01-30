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
from .hydromet_JSON_to_DSS import *

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

def get_volume_region(precip_table_dir: str, vol_col: str='Volume', 
                    reg_col: str='Region', display_print: bool=True) -> list:
    '''Extracts the NOAA Atlas 14 volume and region from the Excel file 
       created by PrecipTable.ipynb
    '''
    df = pd.read_excel(precip_table_dir, sheet_name = 'NOAA_Atlas_MetaData')
    vol =df[vol_col][0]
    reg = df[reg_col][0]
    results = [vol, reg]
    if display_print: print('NOAA Atlas 14: Volume {}, Region {}'.format(vol, reg))
    return results


def get_temporal_map(data_dir: str, filename: str, vol: int, reg: int, 
                                dur: int, display_print: bool=True) -> dict:
    '''Reads the json file containing the temporal distribution data metadata
       and returns the data map and number of rows to skip for the specified
       volume, region, and duration. 
    '''
    with open(data_dir/filename) as json_file:  
        all_map = json.load(json_file)
    sliced_map = all_map[str(vol)]['regions'][str(reg)]['durations'][str(dur)]
    if display_print: print(sliced_map)
    return sliced_map


def get_temporals(temporal_dir: str, vol: int, reg: int, dur: int, 
                    qmap: dict, display_print: bool=True) -> pd.DataFrame:
    '''Reads the csv file containing the temporal distributions for the
       specified volume, region, and duration. Rows with NaNs for an index 
       are dropped. Data was downloaded from:
       https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_temporal.html
    '''
    f = 'Temporals_Volume{0}_Region{1}_Duration{2}.csv'.format(vol, reg, dur)
    path = temporal_dir/f
    s = qmap['skiprows']
    df = pd.read_csv(path, skiprows = s, index_col = 0, keep_default_na=False)
    df = df[df.index!=''].copy()
    for col in df.columns:
        if 'Unnamed' in col:
            del df[col]
    if display_print: print(display(df.head(2)))
    return df


def get_quartile_rank(data_dir: str, filename: str, vol: int, reg: int, 
                                dur: int, display_print: bool=True) -> list:
    '''Extracts the quartile ranks for the specified volume, region, and
       duration. The quartile rank corresponds to the percentage of 
       precipitation events whose temporal distributions are represented
       by those in a specific quartile.
    '''
    input_data = data_dir/filename
    sheet = 'NOAA Atlas 14 Vol {0}'.format(vol)
    df = pd.read_excel(input_data, sheet_name=sheet, index_col=0)
    rank=list(df[(df.index==dur)  & (df['Region']==reg)].values[0])[1:5]
    rank_per = []
    for i in rank: 
        rank_per.append(i/100.0)
    total = sum(rank_per)
    assert 0.99 <= total <= 1.01, 'Sum of ranks not between 99% and 101%' 
    if display_print: print(rank_per)
    return rank_per


def get_duration_weight(data_dir: str, filename: str, vol: int, reg: int, 
                                dur: int, display_print: bool=True) -> list:
    '''Extracts the duration weight for the specified volume, region, and
       duration. The duration weight corresponds to the percentage of 
       precipitation events with the specified duration.
    '''
    input_data = data_dir/filename
    sheet = 'NOAA Atlas 14 Vol {0}'.format(vol)
    df = pd.read_excel(input_data, sheet_name = sheet, index_col=0)
    w=df[(df.index==dur)  & (df['Region']==reg)]['Duration Weight'].values[0]  
    if display_print: print(w)
    return w

def get_quartiles(raw_temporals: pd.DataFrame, dur: int, qrank: list, 
                    qmap: dict, vol: int, reg: int, plot: bool=False) -> dict:
    '''For each quantile, extract the temporal data from the raw_temporals 
       dataframe, convert the data to numeric, store the data in a dictionary, 
       and plot the deciles.
    '''
    idx_name = raw_temporals.index.name
    assert idx_name in ['percent of duration', 'hours'], "Check temporal data"
    curve_group = {}
    for key in qmap['map'].keys():            
        q = raw_temporals[qmap['map'][str(key)][0]:qmap['map'][str(key)][1]].copy()
        if idx_name == 'percent of duration':
            q.index.name = None
            q = q.T
            tstep = dur/(q.shape[0]-1)
            q['hours'] = np.arange(0, dur+tstep, tstep)
        elif idx_name == 'hours':
            q = q.reset_index()
            q['hours'] = pd.to_numeric(q['hours'])
        q = q.set_index('hours')  
        for col in q.columns:
            q[col] = pd.to_numeric(q[col])
        curve_group[key] = q                
    if plot: plot_deciles_by_quartile(curve_group, qrank, qmap, vol, reg, dur)
    return curve_group


def map_quartiles_deciles(n_samples: int=75, seed: int=None, 
                plot: bool=False, display_print: bool=True) -> pd.DataFrame:
    '''Constructs a dataframe containing randomly selected deciles for the 
       specified number of samples (events).
    '''
    if not seed:
        seed = np.random.randint(low=0, high=10000)
    np.random.seed(seed)
    df = pd.DataFrame(index=np.arange(1, n_samples+1))
    df['Deciles'] = np.random.randint(1, 10, n_samples)*10
    if plot: plot_decile_histogram(df)
    if display_print: print('Seed - Deciles:', seed)
    return df


def plot_deciles_by_quartile(curve_group: dict, qrank: list,
                qmap: dict, vol: int, reg: int, dur: int) -> plt.subplots:
    '''Plots the temporal distribution at each decile for each quartile. 
    '''
    fig, ax = plt.subplots(2,2, figsize=(24,10))
    for axi in ax.flat:
        axi.xaxis.set_major_locator(plt.MultipleLocator((
                                            curve_group['q1'].shape[0]-1)/6))
        axi.xaxis.set_minor_locator(plt.MultipleLocator(1))
    axis_num=[[0,0], [0,1], [1,0], [1,1]]
    for i, val in enumerate(qmap['map'].keys()):
        for col in curve_group[val].columns:
            plt.suptitle('Volume '+str(vol)+' Region '+str(reg)+' Duration '+\
                                str(dur), fontsize = 20, x  = 0.507, y = 1.02)
            ax[axis_num[i][0],axis_num[i][1]].plot(curve_group[val][col], 
                                                                label=col) 
            ax[axis_num[i][0],axis_num[i][1]].grid()
            ax[axis_num[i][0],axis_num[i][1]].set_title('Quartile {0}\n{1}%'
                    ' of Cases'.format(i+1, int(qrank[i]*100)), fontsize=16)
            ax[axis_num[i][0],axis_num[i][1]].legend(title='Deciles')
            ax[axis_num[i][0],axis_num[i][1]].set_xlabel('Time (hours)', 
                                                                fontsize=14)
            ax[axis_num[i][0],axis_num[i][1]].set_ylabel('Precip (% Total)', 
                                                                fontsize=14)
    plt.tight_layout()


def plot_decile_histogram(df: pd.DataFrame) -> plt.subplots:
    '''Plots a histogram of the randomly selected decile numbers within the
       passed dataframe.
    '''
    fig = df.hist(bins=20, figsize=(20,6), grid=False)

##need to smooth this per NEH 
def create_inflection_points_nrcs(dur_precip_df_noaa):
    '''
    Docstring for create_inflection_points_nrcs
    
    :param dur_precip_df_noaa: Description
    '''
    dur_precip_df_noaa['ratio'] = dur_precip_df_noaa['value']/dur_precip_df_noaa.loc['24h','value']
    #convert duration to minutes
    min_hr_dict = {'m': 1,'h':60}
    dur_precip_df_noaa['duration_minutes'] = dur_precip_df_noaa.apply(lambda x: int(x.name[:2])*min_hr_dict[x.name[2:]],axis=1)

    #get incremental intensity in/h
    dur_precip_df_noaa['incre_int'] = dur_precip_df_noaa.apply(lambda x: x.value / (x.duration_minutes/60),axis=1)

    #need to add regression here to smooth the 
    #initial_df = dur_precip_df_noaa.copy()[['duration_minutes','incre_int']].set_index('duration_minutes')
    revised_df = dur_precip_df_noaa #smoothed

    #set points along curve from the NEH Handbook. (u)se strings to avoid float issues)
    cumulative_ratio = {}
    cumulative_ratio['0'] = 0
    cumulative_ratio['6'] = 0.5 - (revised_df.loc['12h'].ratio/2)
    cumulative_ratio['9'] = 0.5 - (revised_df.loc['06h'].ratio/2)
    cumulative_ratio['10.5'] = 0.5 - (revised_df.loc['03h'].ratio/2)
    cumulative_ratio['11'] = 0.5 - (revised_df.loc['02h'].ratio/2)
    cumulative_ratio['11.5'] = 0.5 - (revised_df.loc['60m'].ratio/2)
    cumulative_ratio['11.75'] = 0.5 - (revised_df.loc['30m'].ratio/2)
    cumulative_ratio['11.875'] = 0.5 - (revised_df.loc['15m'].ratio/2)
    cumulative_ratio['11.9167'] = 0.5 - (revised_df.loc['10m'].ratio/2)
    return cumulative_ratio

def get_input_data(precip_table_dir: str, duration: int, lower_limit: int=2,
                                 display_print: bool=True) -> pd.DataFrame:
    '''Extracts the precipitation frequency data for the specified duration
       from an Excel sheet and returns the dataframe with the data.  
    '''
    area_precip = 'AreaDepths_{}hr'.format(duration)
    df = pd.read_excel(precip_table_dir, sheet_name= area_precip, index_col=0)
    df_truncated = df[df.index >= lower_limit]
    if display_print: print(display(df_truncated.head(2)))
    return df_truncated

def precip_distributed_nrcs(hydro_events:np.ndarray,nrcs_precip_table_dir: pl.WindowsPath,
                     precip_data: pd.DataFrame, display_print = False):
    """Takes the events, precipitation data, nrcs temporal distribution, CN and applies the CN reduction method to
    obtain a runoff curve for each recurrence interval
    """
    #runoff_distros1 = {}
    prep_curves = pd.DataFrame(columns = hydro_events.astype(float))
    for event in hydro_events:
        dist_df = get_hyeto_input_data_nrcs(nrcs_precip_table_dir, event, display_print)
        # dist_df['precip_cumu'] = dist_df['cu_precip_ratio']*precip_data['Median'].loc[event]
        # dist_df['hyeto_input'] = dist_df['precip_cumu'].diff()
        dist_df['hyeto_input'] = dist_df['precip'].fillna(0.0)
        prep_curves[event] = dist_df['hyeto_input']
        t_step = dist_df.index[1]
    return prep_curves, t_step

def precip_to_runoff_nrcs(hydro_events:np.ndarray,nrcs_precip_table_dir: pl.WindowsPath,
                     precip_data: pd.DataFrame, CN: int, display_print = False):
    """Takes the events, precipitation data, nrcs temporal distribution, CN and applies the CN reduction method to
    obtain a runoff curve for each recurrence interval
    """
    #runoff_distros1 = {}
    prep_curves = pd.DataFrame(columns = hydro_events.astype(float))
    for event in hydro_events:
        dist_df = get_hyeto_input_data_nrcs(nrcs_precip_table_dir, event, display_print)
        s = S_24hr(CN)
        ia = IA_24hr(s)
        #runoff_distros1[event] = excess_precip(dist_df,ia, s)\
        dist_df['precip_cumu'] = dist_df['cu_precip_ratio']*dist_df['precip'].sum()
        dist_df = excess_precip(dist_df,ia, s)
        prep_curves[event] = dist_df['hyeto_input']
        t_step = dist_df.index[1]
    return prep_curves, t_step

def precip_to_runoff_atlas(hydro_events:np.ndarray,atlas14_precip_table_dir: pl.WindowsPath,
                     precip_data: pd.DataFrame, CN: int, quartile:int, display_print = False):
    """Takes the events, precipitation data, nrcs temporal distribution, CN and applies the CN reduction method to
    obtain a runoff curve for each recurrence interval
    """
    #runoff_distros1 = {}
    Atlas14_hyetographs = {1:'q1',2: 'q2',3: 'q3',4: 'q4'}
    prep_curves = pd.DataFrame(columns = hydro_events)
    for event in hydro_events:
        dist_df, weight_df = get_hyeto_input_data_atlas(atlas14_precip_table_dir, Atlas14_hyetographs[quartile], display_print)
        dist_df['precip_cumu'] = dist_df[Atlas14_hyetographs[quartile]]*precip_data['Median'].loc[event]
        s = S_24hr(CN)
        ia = IA_24hr(s)
        dist_df = excess_precip(dist_df,ia, s)
        prep_curves[event] = dist_df['hyeto_input']
        t_step = dist_df.index[1]
    return prep_curves, t_step

def excess_precip(dist_df: pd.DataFrame,ia: float, s: float) -> pd.DataFrame:
    '''Calculates runoff using the curve number approach for a dataframe. See equation 10-9
       of NEH 630, Chapter 10
       (https://www.wcc.nrcs.usda.gov/ftpref/wntsc/H&H/NEHhydrology/ch10.pdf) 
    '''
    dist_df['excess_precip'] = np.where(dist_df['precip_cumu']<= ia, 0, (np.square(dist_df['precip_cumu']-ia))/(dist_df['precip_cumu']-ia+s))
    dist_df['hyeto_input'] = dist_df['excess_precip'].diff()
    dist_df['hyeto_input'] = dist_df['hyeto_input'].fillna(0.0)
    return dist_df

def S_24hr(CN: int) -> float:
    '''Calculates the potential maximum retention after runoff begins (S), in 
       inches.
    '''
    return (1000-10*CN)/CN


def IA_24hr(s24: float) -> float:
    '''Calculats the inital abstraction (Ia) as a function of the maximum
       potentail rention (S). Lim et al. (2006) suggest that a 5% ratio of 
       Ia to S is more appropriate for urbanized areas instead of the more 
       commonly used 20% ratio 
       (https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1752-1688.2006.tb04481.x).
    '''
    return 0.2*s24

def get_hyeto_input_data_nrcs(temporal_precip_table_dir: str, event: int,
                         display_print: bool=True) -> pd.DataFrame:
    '''Extracts the temporal distribution from precipitation frequency data for the specified duration from an Excel 
       sheet and returns the data as a dataframe. 
    '''
    hyeto_precip = 'nrcs_hye_{}'.format(event)
    df = pd.read_excel(temporal_precip_table_dir, sheet_name=hyeto_precip, index_col=0)
    if display_print: 
        print(display(df.head(2)))
    return df

def get_hyeto_input_data_atlas(temporal_precip_table_dir: str, quartile: str,
                         display_print: bool=True) -> tuple:
    '''Extracts the temporal distribution from precipitation frequency data for the specified duration from an Excel 
       sheet and returns the data as a dataframe. 
    '''
    hyeto_precip = 'atlas_hye_{}'.format(quartile)
    df = pd.read_excel(temporal_precip_table_dir, sheet_name=hyeto_precip, index_col=0)
    weights_df = pd.read_excel(temporal_precip_table_dir, sheet_name='atlas_hye_weights', index_col=0)
    if display_print: 
        print(display(df.head(2)))
    return df, weights_df

def extend_time(prep_curves: pd.DataFrame,time_extend: float,time_step: float) -> pd.DataFrame:
    """extends the hyetograph by a select period of time. the timestep is the spacing between
       simulation intervals (typically 0.1 or 0.5 hours)
    """
    extend_curves = prep_curves.loc[0.0:time_extend]*0
    extend_curves.index = extend_curves.index+(24+time_step)
    return pd.concat([prep_curves,extend_curves]).rename_axis('hours')

def weights_noaa(Reccurence_Intervals):
    #Code for making list of years into a list of weights
    weights=[]
    adj_weights = {}
    uni = sorted(list(set(Reccurence_Intervals)))
    for i, year in reversed(list(enumerate(uni))):
        w=round(1/year-sum(weights),9) 
        weights.append(w)
    weights.reverse()
    return weights

def precip_distributed_atlas(hydro_events:np.ndarray,atlas14_precip_table_dir: pl.WindowsPath,
                     precip_data: pd.DataFrame, quartile:int, display_print = False):
    """Takes the events, precipitation data, nrcs temporal distribution, CN and applies the CN reduction method to
    obtain a runoff curve for each recurrence interval
    """
    #runoff_distros1 = {}
    Atlas14_hyetographs = {1:'q1',2: 'q2',3: 'q3',4: 'q4'}
    prep_curves = pd.DataFrame(columns = hydro_events)
    for event in hydro_events:
        dist_df, weight_df = get_hyeto_input_data_atlas(atlas14_precip_table_dir, Atlas14_hyetographs[quartile], display_print)
        dist_df['precip'] = dist_df[Atlas14_hyetographs[quartile]]*precip_data['Median'].loc[event]
        dist_df['hyeto_input'] = dist_df['precip'].diff()
        dist_df['hyeto_input'] = dist_df['precip'].fillna(0.0)
        prep_curves[event] = dist_df['hyeto_input']
        t_step = dist_df.index[1]
    return prep_curves, t_step

def combine_results_stratified(var: str, outputs_dir: str, BCN: str, duration: int, hydrology_IDs: list,
         run_dur_dic: dict=None, remove_ind_dur: bool = True) -> dict:
    '''Combines the excess rainfall *.csv files for each duration into a 
       single dictionary for all durations. A small value of 0.0001 is added so the result is not printed in scientific notation.
    '''
    pd.reset_option('^display.', silent=True)
    assert var in ['Excess_Rainfall', 'Rainfall','Weights'], 'Cannot combine results'
    dic = {}
    df_lst = []
    for ID in hydrology_IDs:
        scen = '{0}_Dur{1}_{2}'.format(BCN, duration, ID)
        file = outputs_dir/'{}_{}.csv'.format(var, scen)
        df = pd.read_csv(file, index_col = 0)
        if var == 'Excess_Rainfall' or var == 'Rainfall':
            df_dic = df.to_dict()
            dates = list(df.index)
            ordin = df.index.name.title()
            events = {}
            for k, v in df_dic.items():
                if 'E' in k or 'N' in k:
                    m = list(v.values())
                    m1= [ float(i)+0.0001 if float(i)< 0.0001  and 0< float(i) else float(i)  for i in m]
                    events[k] = m1
            key ='{0}'.format(str(ID).zfill(2))
            val = {'time_idx_ordinate': ordin, 
                   'run_duration_days': run_dur_dic[str(duration)],
                    'time_idx': dates, 
                    'pluvial_BC_units': 'inch/ts', 
                    'BCName': {BCN: events}}         
            dic[key] = val

        elif var == 'Weights':
            df_lst.append(df)
        if remove_ind_dur:
            os.remove(file)    
    if var == 'Weights':
        all_dfs = pd.concat(df_lst)
        weights_dic = all_dfs.to_dict()
        dic = {'BCName': {BCN: weights_dic['Weight']}}
        #print('Total Weight:', all_dfs['Weight'].sum())
    return dic

def combine_results_traditional(var: str, outputs_dir: str, BCN: str, duration: int, hydrology_IDs: list,
         run_dur_dic: dict=None, remove_ind_dur: bool = True) -> dict:
    '''Combines the excess rainfall *.csv files for each duration into a 
       single dictionary for all durations. A small value of 0.0001 is added so the result is not printed in scientific notation.
    '''
    pd.reset_option('^display.', silent=True)
    assert var in ['Excess_Rainfall', 'Weights'], 'Cannot combine results'
    dic = {}
    df_lst = []
    for ID in hydrology_IDs:
        scen = '{0}_Dur{1}_{2}'.format(BCN, duration, ID)
        file = outputs_dir/'{}_{}.csv'.format(var, scen)
        df = pd.read_csv(file, index_col = 0)
        if var == 'Excess_Rainfall':
            df_dic = df.to_dict()
            dates = list(df.index)
            ordin = df.index.name.title()
            events = {}
            for k, v in df_dic.items():
                m = list(v.values())
                m1= [ float(i)+0.0001 if float(i)< 0.0001  and 0< float(i) else float(i)  for i in m]
                events[k] = m1
            key ='{0}'.format(str(ID).zfill(2))
            val = {'time_idx_ordinate': ordin, 
                   'run_duration_days': run_dur_dic[str(duration)],
                    'time_idx': dates, 
                    'pluvial_BC_units': 'inch/ts', 
                    'BCName': {BCN: events}}         
            dic[key] = val
        elif var == 'Weights':
            df_lst.append(df)
        if remove_ind_dur:
            os.remove(file)    
    if var == 'Weights':
        all_dfs = pd.concat(df_lst)
        weights_dic = all_dfs.to_dict()
        dic = {'BCName': {BCN: weights_dic['Weight']}}
        #print('Total Weight:', all_dfs['Weight'].sum())
    return dic