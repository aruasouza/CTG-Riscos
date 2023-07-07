url = 'https://www.youtube.com/'

import requests
from multiprocessing.dummy import Pool
import time

pool = Pool(1)

def on_success(r):
    if r.status_code == 200:
        print(f'Request succeed: {r}')
    else:
        print(f'Request failed: {r}')

def on_error(ex: Exception):
    print(f'Request failed: {ex}')

before = time.time()

pool.apply_async(requests.get, args=[url],callback=on_success, error_callback=on_error)
# requests.get(url)

print(time.time() - before)

time.sleep(3)