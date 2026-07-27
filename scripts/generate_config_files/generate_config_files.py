import yaml, re
import os, sys
import glob
from os.path import join as pjoin
from pathlib import Path
import traceback

def write_batch_call(out_dir, env_config, configs, mode='process'):

    with open(pjoin(out_dir, 'batch_call.sh'), 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        executable = pjoin('docker', 'SIFMap_runner.sh')
        
        # if we create docker calls
        if env_config is not None:
            for config in configs:
                f.write(f'bash {executable} -c {env_config} {mode} {config} \n')

        else:
            mode += 'ing'
            executable = f'run_UAV_{mode}.py'
            for config in configs:
                f.write(f'python {executable} {config} \n')

def write_new_config_pairs(default_config, default_sensor_calib, calib_dir, raw_data_dir, 
                           out_dir, out_processing_dir, data_dir=None, app_dir=None, user_dir=None, 
                           search_pattern=None):
 
    all_calib_cases, all_raw_dirs, all_subtrees = search_calibration_dirs(calib_dir, raw_data_dir, search_pattern)
    
    all_files = []
    for case_calib, case_raw, case_subtree in zip(all_calib_cases, all_raw_dirs, all_subtrees):
        print('\nWorking on case', case_calib, case_raw, case_subtree)

        case_out = pjoin(out_processing_dir, case_subtree) 
         
        new_config, new_sensor_calib = generate_config_pair(default_config, default_sensor_calib, 
                                                            case_calib, case_raw, case_out, 
                                                            user_dir=user_dir, data_dir=data_dir, app_dir=app_dir)
        
        if new_config is None or new_sensor_calib is None:
            continue

        new_out_dir = pjoin(out_dir, case_subtree)
        os.makedirs(new_out_dir, exist_ok=True)
        
        new_config_file = pjoin(new_out_dir, 'config.yaml')
        new_sensor_calib_file = pjoin(new_out_dir, 'sensor_calib.yaml')

        write_yaml_file(new_config, new_config_file)
        write_yaml_file(new_sensor_calib, new_sensor_calib_file)
        
        if data_dir is not None:
            new_config_file = adjust_path(new_config_file, data_dir, app_dir, user_dir)
            new_sensor_calib_file = adjust_path(new_sensor_calib_file, data_dir, app_dir, user_dir)

        all_files.append((new_config_file, new_sensor_calib_file))

    return all_files

def adjust_path(path, data_dir=None, app_dir=None, user_dir=None):
    replacements = [
        (data_dir, "/data"),
        (app_dir, "/app"),
        (user_dir, "/user"),
    ]

    for old, new in sorted(
        [r for r in replacements if r[0]],
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        if path.startswith(old):
            return path.replace(old.rstrip(os.sep), new, 1)

    return path


def generate_config_pair(default_config_file, default_sensor_calib_file, calib_dir, 
                         raw_data_dir, out_processing_dir, data_dir, app_dir, user_dir):

    new_config = generate_config_file(default_config_file, raw_data_dir, out_processing_dir, 
                                      data_dir, app_dir, user_dir)
    new_sensor_calib = generate_sensor_calib_file(default_sensor_calib_file, calib_dir, 
                                                  raw_data_dir, data_dir, app_dir, user_dir)
    
    if new_config is None or new_sensor_calib is None:
        return None, None

    return new_config, new_sensor_calib


def generate_config_file(default_config_file, raw_data_dir, out_processing_dir, 
                         data_dir, app_dir, user_dir):

    default = load_yaml_file(default_config_file)

    raw_data_dir = adjust_path(raw_data_dir, data_dir, app_dir, user_dir)
    out_processing_dir = adjust_path(out_processing_dir, data_dir, app_dir, user_dir)
    
    default['data_params']['dataset_path'] = raw_data_dir
    default['data_params']['result_path'] = out_processing_dir  
    default['sensor_calibration_config'] = 'sensor_calib.yaml'

    return default
    

def generate_sensor_calib_file(default_sensor_calib_path, calib_dir, raw_data_dir, 
                               data_dir, app_dir, user_dir):
    default = load_yaml_file(default_sensor_calib_path)
    
    flight_nr = int(extract_flight_nr(calib_dir))
    
    try:
        readme = glob.glob(pjoin(os.path.dirname(raw_data_dir), 'ReadMe_*'))[0]
        campaign_info = parse_flight_info(readme)
    
    except Exception as e:
        traceback.print_exc()
        print('\nERROR: Calibration information probably incomplete or not well formatted.')
        return None

    print(campaign_info)
    
    try:
        integration_times = campaign_info['flights'][flight_nr]['sifcam_int_times_ms']
        binning = campaign_info['flights'][flight_nr]['binning'] 

        flat_field_757 = glob.glob(pjoin(calib_dir, 'FFmap757.mat'))
        flat_field_760 = glob.glob(pjoin(calib_dir, 'FFmap760.mat'))
        rad_coeff = glob.glob(pjoin(calib_dir, 'Radiometric Coefficient*'))
        dark_acquisition = glob.glob(pjoin(calib_dir, 'dark*'))
        refl_calib_757 = glob.glob(pjoin(calib_dir, 'MW*757.mat'))
        refl_calib_760 = glob.glob(pjoin(calib_dir, 'MW*760.mat'))

        # logic to change the template
        default['flat_field']['757'] = adjust_path(flat_field_757[0], data_dir, app_dir, user_dir)
        default['flat_field']['760'] = adjust_path(flat_field_760[0], data_dir, app_dir, user_dir)
        
        default['dark_acquisitions'] = adjust_path(dark_acquisition[0], data_dir, app_dir, user_dir)

        default['reflectance_calibration']['757']['panel_measurements'] = adjust_path(refl_calib_757[0], data_dir, app_dir, user_dir)
        default['reflectance_calibration']['760']['panel_measurements'] = adjust_path(refl_calib_760[0], data_dir, app_dir, user_dir)
        
        default['radiance_calibration'] = adjust_path(rad_coeff[0], data_dir, app_dir, user_dir)
        
        default['integration_times']['760'] = integration_times['760']
        default['integration_times']['757'] = integration_times['757']

        if binning == '2x2':
            kmat = [[1923, 0, 512], 
                    [0, 1923, 512], 
                    [0, 0, 1]]

        elif binning == '4x4':
            kmat = [[962, 0, 256], 
                    [0, 962, 256], 
                    [0, 0, 1]]
        
        else:
            raise NotImplemtentedError()
        
        default['KMat'] = kmat
        default['mask'] = None
    
    except Exception as e:
        traceback.print_exc()
        print(f'\nERROR: Calibration file ({readme}) not well formatted.')
        return None

    return default


def extract_flight_nr(path):
    # Regular expression to search for "flight" in the file name
    pattern = r'/flight(\d+)(?:/|$)'
    
    # Use regex to find the flight number
    match = re.search(pattern, path.lower())
    if match:
        return match.group(1)
    else:
        return None


def search_calibration_dirs(calib_dir, raw_data_dir, search_pth=None):
    all_dark_files = glob.glob(pjoin(calib_dir, '**', 'dark*'), recursive=True)
    all_calib_dirs = [os.path.dirname(p) for p in all_dark_files]
    
    c_dirs, r_dirs, s_dirs = [], [], []
    for calib in all_calib_dirs:
        subtree = calib[len(calib_dir.rstrip(os.sep)) + 1:]
        raw = pjoin(raw_data_dir, subtree)
            
        if search_pth is not None:
            raw += search_pth
        
        raw = pjoin(raw, '**', '*light*.tif')
        raw = glob.glob(raw, recursive=True)
        raw = raw[0] if len(raw) > 0 else None
        
        raw = os.path.dirname(raw) if raw is not None and os.path.exists(raw) else None

        if raw is not None:
            subtree = raw[len(raw_data_dir.rstrip(os.sep)) + 1:]
            r_dirs.append(raw)
            c_dirs.append(calib)
            s_dirs.append(subtree)

        else:
            print(f"\nWARNING: could not find a valid raw_data path to the calib_dir {calib}")

    return c_dirs, r_dirs, s_dirs 
    

def load_yaml_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None

def write_yaml_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, width=100000)


def parse_flight_info(filename):
    text = Path(filename).read_text(encoding="utf-8")

    result = {
        "day": None,
        "people": {},
        "flights": {},
    }

    # Day
    m = re.search(r"Day:\s*(.+)", text)
    if m:
        result["day"] = m.group(1).strip()

    # People section
    people_match = re.search(
        r"People:\s*(.*?)\n\s*Flight\s+\d+:",
        text,
        re.DOTALL,
    )

    if people_match:
        people_text = people_match.group(1)

        for line in people_text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue

            name, value = line.split(":", 1)
            result["people"][name.strip()] = value.strip()

    # Flights
    flight_pattern = re.compile(
        r"Flight\s+(\d+):\s*"
        r"(\d{2}:\d{2})-(\d{2}:\d{2})"
        r"(?:\s*\(([^)]*)\))?\s*"
        r"(.*?)(?=\n\s*Flight\s+\d+:|\Z)",
        re.DOTALL,
    )

    for match in flight_pattern.finditer(text):
        number = int(match.group(1))
        start_time = match.group(2)
        end_time = match.group(3)
        log_file = match.group(4).strip() if match.group(4) is not None else None
        body = match.group(5)

        flight = {
            "start_time": start_time,
            "end_time": end_time,
            "log_file": log_file,
        }

        # Altitude
        m = re.search(r"Altitude:\s*(\d+)\s*m", body)
        if m:
            flight["altitude_m"] = int(m.group(1))

        # SIFcam integration times and binning
        m = re.search(
            r"SIFcam int\. times:\s*(\d+)\+(\d+)\s*ms,\s*binning\s*([0-9x]+)",
            body,
        )
        if m:
            flight["sifcam_int_times_ms"] = [
                int(m.group(1)),
                int(m.group(2)),
            ]

            # format is (int.time 757, int. time 760)
            flight['sifcam_int_times_ms'] = {'757':flight['sifcam_int_times_ms'][0],'760':flight['sifcam_int_times_ms'][1]}
            flight["binning"] = m.group(3)

        # Log availability
        lower_body = body.lower()

        flight["drone_log_available"] = (
            "drone available" in lower_body
            or "drone is available" in lower_body
            or "log file from the drone available" in lower_body
        )

        flight["gimbal_log_available"] = not (
            "gimbal is not available" in lower_body
            or "log file from the gimbal is not available" in lower_body
        )

        result["flights"][number] = flight
    
    return result


def main():
    # To capture command-line arguments, we can use the `sys.argv` list.
    if len(sys.argv) != 12:
        print("Usage: python process_data.py default_config default_sensor_calib calib_dir raw_data_dir output_dir output_processing_dir data_dir user_dir app_dir env_config pattern")
        sys.exit(1)

    default_config = sys.argv[1]
    default_sensor_calib = sys.argv[2]
    calib_dir = sys.argv[3]
    raw_data_dir = sys.argv[4]
    out_path = sys.argv[5]
    out_processing_dir = sys.argv[6]
    data_dir = sys.argv[7]
    user_dir = sys.argv[8]
    app_dir = sys.argv[9]
    env_config = sys.argv[10]
    search_pattern = sys.argv[11]

    if user_dir == 'None':
        user_dir = None

    if app_dir == 'None':
        app_dir = None

    if data_dir == 'None':
        data_dir = None

    if env_config == 'None':
        env_config = None

    all_files = write_new_config_pairs(
        default_config,
        default_sensor_calib,
        calib_dir,
        raw_data_dir,
        out_path,
        out_processing_dir, 
        data_dir=data_dir, 
        user_dir=user_dir, 
        search_pattern=search_pattern
    )
    
    #from pathlib import Path
    #package_dir = (Path(__file__) / '..' / '..').resolve()
    write_batch_call(out_path, env_config, [p[0] for p in all_files])

if __name__ == "__main__":
    main()

