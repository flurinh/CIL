import numpy as np
from io import BytesIO
import time
import PIL
from bs4 import BeautifulSoup
import requests
from PIL import Image
import PIL
import random
import glob
import os
import imageio

def get_soup(link=None):
    rls = requests.get(link)
    soup = BeautifulSoup(rls.content, 'html.parser')
    return soup

def find_img(soup):
    return soup.findAll("hrev")

def download(link, folder):
    print('Downloading', link)
    rq = requests.get(link, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:63.0) Gecko/20100101 Firefox/63.0'})
    file=folder+'/'+link[-16:-8]+'.png'
    if rq.status_code == 200:
        img1 = Image.open(BytesIO(rq.content))
        img2 = imageio.imread(BytesIO(rq.content))
        print("datashape: "+str(img2.shape))
        img1.save(file)
        
def dl(url, training_folder, waittime=1, start = 10078660, end = None):
    link_list=[]
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    dl = False
    for element in soup.findAll('a', href=True):
        link = element['href']
        if str(start) in link:
            dl = True
        if str(end) in link:
            break
        if dl & ('15.tif' in link):
            link_list.append(link)
    for l in range(len(link_list)):
        if 'map' in url:
            'Downloading groundtruth maps...'
            folder = training_folder + 'map_data'
        if 'sat' in url:
            'Downloading satellite images...'
            folder = training_folder + 'sat_data'
        download(link_list[l], folder)
        time.sleep(waittime)
        
url_map = 'http://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html'
url_sat = 'http://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html'

dl(url_map, training_folder = 'training/')
dl(url_sat, training_folder = 'training/')