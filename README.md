# ANNNI Phase detection with MPS States

### How to install
1. Create a virtual environment: 
```bash
python3 -m venv annni
```

2. Activate it:
```bash
source annni/bin/activate
```

3. Install the requirements:
```bash
pip3 install .
```

### How to get the states?
1. Copy the URL of the Drive folder:
   ```bash
   https://drive.google.com/drive/u/1/folders/******
   ```
2. Run the python script passing the URL with the `--url` flag
   ```bash
   python getdata.py --url https://drive.google.com/drive/u/1/folders/******
   ```
   
### How to generate new states? 
Use the script [getstates.py](src/ANNNI/scripts/dmrg/getstates.py)