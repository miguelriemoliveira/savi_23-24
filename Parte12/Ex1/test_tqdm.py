#!/usr/bin/env python3


from time import sleep
from tqdm import tqdm


names = ['miguel', 'ze', 'bruno', 'carolina']

for name in tqdm(names, total=len(names), desc='Iterating names'):
    # print(name)
    sleep(1)
