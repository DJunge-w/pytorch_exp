## Install
1. create a virtual environment named `.venv`.\
`python3 -m venv .venv`
2. Activate the virtual environment.\
`source .venv/bin/activate`
3. Install the requirments.\
`pip install -r requirements.txt`

## Run 
1. Activate the virtual environment.\
`source .venv/bin/activate`
2. Run experiment with config file.\
`python run_exp.py -f 1model.json`
3. read loggings.\
`tail -f logs/model_A.log`