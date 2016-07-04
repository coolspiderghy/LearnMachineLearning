import os
import zipfile
import requests
OCCUPANCY = ('http://archive.ics.uci.edu/ml/machine-learning-databases/00357','occupancy_data.zip')
#OCCUPANCY = ('http://bit.ly/ddl-occupancy-dataset', 'occupancy.zip')
#CREDIT    = ('http://bit.ly/ddl-credit-dataset', 'credit.xls')
#CONCRETE  = ('http://bit.ly/ddl-concrete-data', 'concrete.xls')
def download_data(url, name, path='data'):
    if not os.path.exists(path):
        os.mkdir(path)

    response = requests.get(url)
    with open(os.path.join(path, name), 'w') as f:
        f.write(response.content)

def download_all(path='data'):
    #for  oc_item in (OCCUPANCY):#, CREDIT, CONCRETE):
    href, name=OCCUPANCY[0],OCCUPANCY[1]
    print href,name
    download_data(href, name, path)

    # Extract the occupancy zip data
    z = zipfile.ZipFile(os.path.join(path, 'occupancy_data.zip'))
    z.extractall(os.path.join(path, 'occupancy_data'))

path='data'
download_all(path)