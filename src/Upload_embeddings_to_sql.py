# coding: utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os
import re
import sqlite3
from itertools import combinations, combinations_with_replacement

os.chdir('..')

networkdb = sys.argv[2]
con = sqlite3.connect(os.path.join('data',networkdb))

start = 1959*12+9
end = 2016*12+4
time_var = 'month'
agreement_metric = 'agreement'
location='senate'
plt.rcParams['figure.figsize'] = 16,9

congressdb = sys.argv[1]
congress_con = sqlite3.connect(os.path.join('data',congressdb))


# This computes embeddings for each time slice in a sequence, with dependency between them -
# the prior embedding is used as a seed for the next embedding where the politician id's overlap
def plot_embed_upload(networks_con,congress_con,time_var,start,end,location,agreement_metric='agreement',threshold=0.2,xlim=None,ylim=None,savefig=False,savedir=None,upload=False):
    xy=None
    con = networks_con
    
    senators = pd.read_sql_query('select * from senators',congresscon)
    senators.set_index('id',inplace=True)

    for time in np.arange(start,end+1):
        sims = get_distances(con=con,agreement_metric=agreement_metric,time_var=time_var,time=time,location=location,transform=None)
        if sims is None:
            continue
        
        if xy is not None:
            xy, ids, centers = compute_embedding(sims=sims,last_ids=ids,last_embedding=xy,last_centers=centers,senators=senators)
        else:
            xy, ids, centers = compute_embedding(sims=sims,senators=senators)
        
        if time_var == 'month':
            year = time//12
            month = time%12
            if month == 0:
                month = 12
                year -= 1
        elif time_var == 'quarter':
            year = time//4
            quarter = (time%4)
            if quarter == 0:
                quarter = 4
                year -= 1
            month = quarter * 3
        elif time_var == 'year':
            year = time
            month = ""
        
        if xlim is not None and ylim is not None:
            plot_network(sims,xy,threshold=threshold,xlim=xlim,ylim=ylim,location=location,month=month,year=year,senators=senators)
            plt.axis('off')
        else:
            plot_network(sims,xy,threshold=threshold,location=location,month=month,year=year,senators=senators)
        
        if upload:
            upload_embedding(con,ids=ids,embedding=xy,time_var=time_var,time=time,location=location)
            
        if savefig:
            plt.savefig(os.path.join(savedir,'{}_{}_{}.png'.format(location,time_var,time)))
            
        plt.close()
        
        
        
plot_embed_upload(networks_con=con,congress_con=congress_con,time_var=time_var,start=start,end=end,location=location,
                      agreement_metric=agreement_metric,
                      threshold=0.1,xlim=(-0.75,0.75),ylim=(-0.5,0.5),
                      upload=True,savefig=True,savedir='plots')

