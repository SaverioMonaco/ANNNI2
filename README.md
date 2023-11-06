# ANNNI Phase detection with MPS States

### How to install
1. Create a virtual environment: 
```bash
python3 -m venv env
```

2. Activate it:
```bash
source env/bin/activate
```

3. Install the requirements:
```bash
pip3 install -r requirements.txt
```

4. If you get errors from jax and jaxlib, install them using
```bash
pip install -U "jax[cuda12_pip]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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
   