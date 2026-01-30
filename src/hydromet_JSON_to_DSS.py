from .core import*
from datetime import datetime
import shutil
geoDF = 'GeoDataFrame'
plib = 'pathlib.Path'

#---------------------------------------------------------------------------#
def main(forcing_dir: plib, outputs_dir: plib, bin_dir: plib, filename: str, 
                jsonname: str = '', variable: str="Excess-Rainfall", data_type: str='INST-VAL', 
                            scaling: bool=True, units: str='INCHES', remove_temp_files: bool=True, 
                                            display_print: bool=True) -> None:
    '''For each JSON within the forcing directory, the function extracts the
       excess precipitation data for each boundary condition and duration 
       within a given domain's JSON file and saves the data to a single DSS 
       file.

       Parameters
       ----------
       forcing_dir: The path of the forcing directory containing the JSON 
                    files which will be converted to DSS. 
       outputs_dir: The path to the directory where the final DSS file is 
                    saved. 
       bin_dir: The path to the binary directory which contains the dssutl 
                executable. 
       filename: The name of the directory containing the forcing data, as 
                 a string.
       jsonname: the name of the forcing data, as 
                 a string
       variable: A description of the data representation, as a string, i.e. 
                'Excess-Rainfall'.
       data_type: The type of data, as a string, e.g. 'INST-VAL' corresponds to
                  instantaneous value. 
       scaling: yes if being utilized for a stepped hydrology approach, i.e.
                the scaling 2dble project
       units: The units, as a string, of the excess rainfall which will be 
              specified within the final DSS file. 
       remove_temp_files: Bool specifying whether to remove the intermediate 
                          input files generated during the construction of 
                          the final DSS file. 
       display_print: Bool specifying whether to display print statements.     

       Returns
       -------
       None

    '''
    files = []
    temp_files = ['DSSUTL.EXE', 'DSS_MAP.input']
    all_plans = []
    if scaling is False:
        for file in forcing_dir.glob('*.json'):
            files.append(file)
    else:
        files.append(pl.Path(forcing_dir)/jsonname)
    for file in files:
        if display_print:
            print('Converting {} to DSS...'.format(file.name))
        with open(file) as f:
             data = json.load(f)
        plans = list(data.keys())   
        for plan in plans:
            idx_ord = data[plan]['time_idx_ordinate'].lower()
            idx = data[plan]['time_idx']
            BCNames = list(data[plan]['BCName'].keys())
            for BCN in BCNames:
                if 'D' in BCN:
                    pluv_domain = BCN  
            for BCN in BCNames:
                if 'D' in BCN:
                    scen_name = '{0}_{1}'.format(pluv_domain, plan)
                elif 'L' in BCN:
                    scen_name = '{0}_{1}_{2}'.format(pluv_domain, BCN, plan)                    
                else:
                    if scaling is True:
                        scen_name = '{0}_{1}'.format(BCN, plan)
                    else:
                        print(BCN, 'domain type not supported') 
                df = pd.DataFrame.from_dict(data[plan]['BCName'][BCN])
                df[idx_ord] = idx
                df = df.set_index(idx_ord)
                tstep_dic = determine_tstep_units(df)
                tstep = list(tstep_dic.keys())[0]
                tstep_units = list(tstep_dic.values())[0]
                to_dss = 'ToDSS_{0}.input'.format(plan)
                if plan not in all_plans:
                    all_plans.append(plan)
                    dss_map(outputs_dir, variable, tstep, tstep_units, units,
                                     data_type, to_dss = to_dss, open_op = 'a+')
                    temp_files.append(to_dss)
                    excess_df_to_input(outputs_dir, df, tstep, tstep_units, 
                                                        scen_name, 'a+', to_dss) 

            if scaling is True:
                make_dss_file(outputs_dir, bin_dir, '{0}_{1}'.format(filename,plan), to_dss = to_dss, remove_temp_files = False, 
                                                display_print = display_print)
                temp_files.remove(to_dss)
                os.remove(outputs_dir/to_dss)

    if scaling is False:
        make_dss_file(outputs_dir, bin_dir, filename, remove_temp_files = False, 
                                                display_print = display_print)
            
    if remove_temp_files:
        for file in temp_files:
            os.remove(outputs_dir/file)       
    return 

if __name__== "__main__":
        main()


#---------------------------------------------------------------------------#    

def determine_tstep_units(incr_excess: pd.DataFrame) -> dict:
    '''Determines the timestep and the timestep's units of the incremental
       excess runoff.
    '''
    assert incr_excess.index.name == 'hours', 'Timestep and timesteps units' 
    'cannot be calculated if the runoff duration is not in units of hours'
    tstep = incr_excess.index[-1]/(incr_excess.shape[0]-1)
    dic = {}
    if tstep < 1.0:        
        dic[int(60.0*tstep)] = 'MIN' 
    elif tstep >= 1.0:
        dic[int(tstep)] = 'HOUR'
    else:
        print('Timestep and timestep units were not determined')
    return dic


def dss_map(outputs_dir: str, var: str, tstep: int, tstep_units: str,
                units: str, dtype: str='INST-VAL', IMP: str='DSS_MAP.input', 
                        to_dss: str='ToDSS.input', open_op: str='w') -> None:
    '''Creates a map file containing the data structure for DSSUTL.EXE.
    '''
    var8 = var[:8]
    ts = '{0}{1}'.format(tstep, tstep_units)
    output_file = outputs_dir/IMP
    datastring = "EV {0}=///{0}//{1}// UNITS={2} TYPE={3}\nEF [APART] [BPART] [DATE] [TIME] [{0}]\nIMP {4}\n".format(var8, ts, units, dtype, to_dss)
    with open(output_file, open_op) as f: 
        f.write(datastring)
    return None

def excess_df_to_input(outputs_dir: str, df: pd.DataFrame, tstep: float,
                        tstep_units: str, scen_name: str, open_op: str='w', 
                                        to_dss: str='ToDSS.input') -> None:
    '''Writes the excess rainfall dataframe to an input file according to the 
       struture specified within DSS_MAP.input.
    '''
    temp_data_file = outputs_dir/to_dss
    cols = df.columns.tolist()
    start_date = datetime(2021, 1, 1, 1, 00)
    with open(temp_data_file, open_op) as f:
        for i, col in enumerate(cols):
            m_dtm = start_date
            event_data = df[col]
            for j, idx in enumerate(event_data.index):
                if j > 0: 
                    if tstep_units == 'MIN': 
                        m_dtm+=pd.Timedelta(minutes = tstep)
                    elif tstep_units == 'HOUR': 
                        m_dtm+=pd.Timedelta(hours = tstep)
                htime_string = datetime.strftime(m_dtm, '%d%b%Y %H%M')
                runoff = event_data.loc[idx]
                f.write('"{}"'.format(scen_name)+' '+col+' '+htime_string+' '+str(runoff)+'\n')
    return None


def make_dss_file(outputs_dir: str, bin_dir: str, dss_filename: str,
                        dssutil: str='DSSUTL.EXE', IMP: str='DSS_MAP.input', 
                    to_dss: str='ToDSS.input', remove_temp_files: bool=True, 
                                        display_print: bool = True) -> None:
    '''Runs the DSSUTL executable using the DSS_MAP.input file to map the 
       excess rainfall data from the ToDSS.input file and saves the results
       to a dss file.
    '''
    cwd = os.getcwd()
    shutil.copy(bin_dir/dssutil, outputs_dir)
    os.chdir(outputs_dir)
    os.system("{0} {1}.dss INPUT={2}".format(dssutil, dss_filename, IMP))
    time.sleep(5)
    if remove_temp_files:
        os.remove(outputs_dir/IMP)
        os.remove(outputs_dir/dssutil)
        os.remove(outputs_dir/to_dss)
    filepath = outputs_dir/dss_filename
    os.chdir(cwd)
    if display_print: print('Dss File written to {0}.dss'.format(filepath))
    return None
