import os, gdown, argparse

parser = argparse.ArgumentParser(description='Script for fetching the MPS states from Drive')

parser.add_argument('--url', type=str, default='', metavar='URL',
                    help='URL of the Drive folder')

URL = parser.parse_args().url
if len(URL) == 0:
    raise SyntaxError('Provide a valid URL using --url flag')

# if not os.path.exists('./tensor_data/'):
#     os.makedirs('./tensor_data')

print('(1/2) Downloading the data')

gdown.download_folder(URL, quiet=True)

print('(2/2) Extracting all folders')

for file in os.listdir('./tensor_data'):
    # Extract the folder
    os.system(f'tar -xf ./tensor_data/{file} -C ./tensor_data/')
    # Remove the compressed one
    os.system(f'rm ./tensor_data/{file}')