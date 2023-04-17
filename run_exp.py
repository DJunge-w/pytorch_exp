import argparse
import collections
import json
import subprocess
import sys
import os
import tempfile
from time import sleep

def createModelTempFile(models):
    model_files = []
    for model in models:
        # Create temporary file
        model_file = tempfile.NamedTemporaryFile(mode='w+', prefix=f"{model['model_name']}_", suffix=".json") 
        # Write model to temporary file
        json.dump(model, model_file, indent=4)
        model_file.flush()
        model_files.append(model_file)
    return model_files

def destroyModelTempFile(model_files):
    for model_file in model_files:
        model_file.close()

def main():
    # Get input file path
    parser = argparse.ArgumentParser(description="Core GPU Sharing Experiment")
    parser.add_argument('-f', '--file', metavar="FILEPATH", help="Specifies the path to the experiment configuration file", required=True)
    args = parser.parse_args()
    filename = args.file

    # Parse Experiment Configuration JSON file
    try:
        with open(filename, 'r') as file_input:
            experiment_config = json.load(file_input, object_pairs_hook=collections.OrderedDict)
    except FileNotFoundError:
        print(f"Input Experiment Config file: [{filename}] not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Input Experiment Config file: [{filename}] invalid.", file=sys.stderr)
        sys.exit(1)
    
    #Run each model
    models = experiment_config.get('models', [])
    model_pids = []
    model_processes = []
    log_files = []
    model_files = createModelTempFile(models)
    for i, model_file in enumerate(model_files):
        # Run src/run_model.py with current model configuration
        model = models[i]
        model_name = model['model_name']
        # logging 
        logfilename = f"{model['output_file_path']}/{model['output_file_name']}.log"
        os.makedirs(os.path.dirname(logfilename), exist_ok=True)
        logfile = open(f"{model['output_file_path']}/{model['output_file_name']}.log", "w") 
        log_files.append(logfile)
        #run each model as a process
        model_process = subprocess.Popen(['python', 'src/run_model.py', model_file.name, str(experiment_config.get('device_id'))], stdout=logfile, stderr=logfile)
        #record model's PID
        model_pids.append((model_name, model_process.pid))
        model_processes.append(model_process)
    
    #Stop each model at given experiment duration exhausted 
    exp_duration = experiment_config.get('exp_dur')
    sleep(exp_duration)
    for i, p in enumerate(model_processes):
        # p.wait()
        logfile = log_files[i]
        logfile.flush()
        p.kill()
        logfile.close()
    destroyModelTempFile(model_files)
    
    # Write out.log
     
    with open('models_pid.json', 'w') as out_log:
        json.dump(model_pids, out_log, indent=4)
        out_log.flush()
        print("PID summary log saved as [models_pid.json]")


if __name__ == '__main__':
    main()
