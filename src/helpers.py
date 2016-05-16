#encoding:utf-8
import json
import os
import re
import logging
import sqlite3
import pandas as pd
import numpy as np
from lxml import objectify
from functools import reduce
from itertools import product, combinations_with_replacement, combinations

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA


####################################################
## sqlite helper functions
####################################################

# can just execute init_sqlite_db("path/to/database.db", read_schema("path/to/schema/folder"))
# to create the db

legalConfigExtensions = {'.json','.yml'}
sqlite_types = {'TEXT','INTEGER','REAL','BLOB'}


def init_sqlite_db(db_path,schema,init_index=False,index_suffix=''):
    con = sqlite3.connect(db_path)
    
    for table, table_info in schema.items():
        create_table(con,init_index=init_index,index_suffix=index_suffix,**table_info)
        
    con.close()
    

def read_schema(schema_dir,extensions=['.yml','.json'],legal_types=sqlite_types):
    tables = [name for name in os.listdir(schema_dir) if os.path.splitext(name)[1] in extensions]
    db_schema = {}
    for filename in tables:
        table_info = read_table_config(os.path.join(schema_dir,filename),legal_types=legal_types)
        db_schema[table_info['name']] = table_info
        
    return db_schema


def read_table_config(table_path,legal_types=sqlite_types):
    table_info = load_config(table_path)
    table = table_info.get('name',os.path.splitext(os.path.basename(table_path))[0])
    keys = table_info.get('keys',None)
    index = table_info.get('index',None)
    schema = reduce(lambda l1,l2: l1+l2, table_info['schema'],[])
    datatypes = schema[1::2]
    fields = schema[0::2]
    for key in keys:
        if key not in fields:
            raise ValueError("Error in table config {}: key {} not in schema".format(filename,key))
    if index:
        for field in index:
            if field not in fields:
                raise ValueError("Error in table config {}: index field {} not in schema".format(filename,field))
    for datatype in datatypes:
        if datatype not in legal_types:
            raise ValueError("Error in table config {}: datatype {} not in legal_types: {}".format(filename,datatype,legal_types))
    
    table_info.update({'name':table,'fields':fields,'types':datatypes})
    
    return table_info

        
def create_table(con,init_index=False,index_suffix='',**table_info):
    """
    con is a sqlite3 database connection object
    init_index: boolean.  Create an index now if one is specified in table_info?
        it may be better to wait until the table is built, and then call init_index(con,**table_info)
    if_exists: sqlite keyword- what to do if the table already exists. {'ignore','replace','rollback','abort','fail'}
    table_info can be passed as a dict of keyword args.
        Most robust method is to read this from a table config with read_table_config()
        Implicit keyword args are:
        - fields: list of field names in the table
        - keys: list of field names to serve as the primary key
        - types: list of datatypes as strings
        - index: list of field names to be used optionally for creating an index for fast lookup
    """
    keys = table_info.get('keys',None)
    schema = reduce(lambda l1,l2: l1 + l2, table_info['schema'], [])
    name = table_info['name']
    command = ("create table if not exists {} (" + "{} {}, "*(len(schema)//2) + "PRIMARY KEY (" + ','.join(keys) + ") )").format(name,*schema)
    cur = con.cursor()
    cur.execute(command)
    con.commit()
    cur.close()
    
    if init_index:
        if 'indices' not in table_info:
            print("Warning: init_index is True but no index is specified for table {}".format(name))
        else:
            create_index(con,index_suffix,**table_info)


def create_index(con,index_suffix='',**table_info):
    indices = table_info['indices']
    name = table_info['name']
    cur = con.cursor()
    for index_name, index in indices.items():
        command = "create index if not exists {}_{}{} on {}(".format(name,index_name,index_suffix,name) + ','.join(index) + ")"
        cur.execute(command)
    con.commit()
    cur.close()
 
    
def insert_rows(cur,table,data,fields,how='replace'):
    command = ("insert or {} into {} VALUES ("+','.join(["?"]*len(fields))+")").format(how,table)
    cur.executemany(command,data)


def load_config(path, module_to_dict=True):
    """
    loads configuration from .json, .yml, and .py.
    always returns a dict, even in the .py case by default; it's up to you how you'd like to
    handle the result, e.g. as-is or using  globals.update(config) to get direct
    variable name access (only works if the result is a dict, which may not be the
    case for example with some .json files defining list structures at the top level.
    if module_to_dict is True (default), python modules are unpacked into an explicit dict.
    """

    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[-1]
    if ext not in legalConfigExtensions:
        raise ValueError("{} is not a currently supported extension."+
                        "Supported file types are: {}".format(ext,legalConfigExtensions))

    if ext=='.yml':
        from yaml import safe_load as load_yaml
        with open(path,'r') as infile:
            config = load_yaml(infile)
    elif ext=='.json':
        from json import load as load_json
        with open(path,'r') as infile:
            config = load_json(infile)
    elif ext=='.py':
        directory = os.path.split(path)[0]
        module = os.path.splitext(filename)[0]
        cwd = os.getcwd()
        if directory != '':
            os.chdir(directory)
        config_module = import_module(module)
        os.chdir(cwd)
        if module_to_dict:
            names = [name for name in dir(config_module) if not name.startswith('__')]
            config = dict()
            for name in names:
                config[name] = config_module.__dict__[name]
        else:
            config = config_module

    return config



######################################################
## Helpers for Import_govtrack_to_sql.py
######################################################

congress_num = re.compile('[0-9]{1,3}')
congress_bill = re.compile('(s|h)[0-9]{1,5}')
date_regex = re.compile(r'([0-9]{4}(-|/)[0-9]{1,2}\2[0-9]{1,2})')
typedict = {'TEXT':str,'INTEGER':int}

def get_bill_data(root):
    # extract all the data on a single bill from the root xml object
    datetime = pd.to_datetime(root.attrib['datetime'])
    year = int(datetime.year)
    quarter = 4*year+int(datetime.quarter)
    month = 12*year+int(datetime.month)
    
    counts = [int(get_attr(root,kind,0)) for kind in ['aye','nay','present','nv']]
    text_attrs = [get_node_text(root,tag,'') for tag in ['required','result','category','type','question']]
    
    return [root.attrib['where'],int(root.attrib['session']),int(root.attrib['roll']),
            year, quarter, month] + counts + text_attrs
    
def get_attr(obj,attr,default):
    return obj.attrib.get(attr,default)

def get_node_text(obj,tag,default):
    return obj.__dict__.get(tag,objectify.StringElement(default)).text
    
def type_or_null(i,datatype):
    if pd.isnull(i):
        return None
    else:
        return datatype(i)

def year_quarter_month(datetime):
    dt = pd.to_datetime(datetime)
    return pd.Series([dt.year,4*dt.year+dt.quarter,12*dt.year+dt.month])
    
    

######################################################
## Helpers for Extract_networks_to_sql.py
######################################################

vote_map = {'+':1,'-':-1,'0':0,'P':0,'NAY':-1,'YEA':1,None:0,np.nan:0}

def vote_table(con,time_col,time,location,lag=0,drop_dups=True):
    # get all votes for a certain time frame, with bills as columns and voters as index
    start = time - lag
    end = time
    command = """
        select session, year, roll, votes.id, party, vote
        from votes join senators on votes.id = senators.id
        where {} >= {} and {} <= {} and location = '{}'""".format(time_col,start,time_col,end,location)
    
    data = pd.read_sql_query(command,con)
    
    num_rows = data.shape[0]
    
    if drop_dups:
        data.drop_duplicates(subset=['session','year','roll','id','party'],keep='last',inplace=True)
        # report if duplicates were dropped
        if data.shape[0] < num_rows:
            print("{} duplicated entries for {} {}, {}".format(num_rows-data.shape[0],time_col,time,location))
            print("{} remaining rows after duplicate removal".format(data.shape[0]))
    
    data.vote = data.vote.apply(lambda s: vote_map.get(s,0))
    
    data = data.set_index(['id','party','session','year','roll']).unstack(['session','year','roll'])
    data.fillna(0,inplace=True)
    data.columns = data.columns.droplevel(0)
    return data

def vote_dist_series(votes):
    votes,vote_counts = np.unique(votes,return_counts=True)
    vote_counts = dict(zip(votes,vote_counts/vote_counts.sum()))
    if len(vote_counts)<3:
        vote_counts[-1] = vote_counts.get(-1,0)
        vote_counts[0] = vote_counts.get(0,0)
        vote_counts[1] = vote_counts.get(1,0)
    return pd.Series(vote_counts)

def vote_entropy(vote_dist):
    return entropy(vote_dist.values)

def voter_agreement(voter1,voter2,probs=None,weights=None):
    return np.mean(voter1.values*voter2.values)
    #return np.logical_and(voter1==voter2,voter1!=0).values.mean()

def voter_agreement_pos(voter1,voter2,probs=None,weights=None):
    #agreements = (voter1.values*voter2.values)
    #agreements = agreements[agreements>0]
    locs=voter1*voter2 != 0
    if locs.sum() == 0:
        return 0.0
    return (voter1==voter2)[locs].mean()
    
def voter_perplexity(voter1,voter2,probs,weights=None):
    v1 = voter1.values#.astype('float')
    v2 = voter2.values#.astype('float')
    p1 = probs.values[v1.astype('int')+1,np.arange(probs.shape[1])]#.astype('float')
    p2 = probs.values[v2.astype('int')+1,np.arange(probs.shape[1])]#.astype('float')
    
    perp = -1*(np.log(p1)+np.log(p2))*v1*v2
    p = (np.sum(perp))
    count = len(np.nonzero(v1*v2)[0])
    if count > 0:
        return p/count
    else:
        return 0.0

def affinity_table(vote_table,weight_func=None,affinity_func=voter_agreement_pos,transform=None):
    vote_dist = vote_table.apply(vote_dist_series,axis=0)
    vote_dist.sort_index(axis=0,inplace=True)
    
    if weight_func is not None:
        weights = vote_dist.apply(weight_func,axis=0)
    else:
        weights = None
        
    affinities = pd.DataFrame(index = vote_table.index,columns = vote_table.index,dtype='float')
    
    for voter in vote_table.index:
        affinities.loc[voter,:] = vote_table.apply(affinity_func,axis=1,voter2=vote_table.loc[voter,],
                                                   probs=vote_dist,weights=weights).values
    if transform:
        affinities = affinities.applymap(transform)
        
    return affinities

def interparty_affinity(affinity_table,summary_func=np.mean):
    min_id = np.min(affinity_table.index.levels[0].values)
    max_id = np.max(affinity_table.index.levels[0].values)
    all_ids = slice(min_id,max_id)
    
    parties = affinity_table.index.levels[1].unique()
    affinities = pd.DataFrame(index=parties,columns=parties,dtype = 'float')
    
    for pair in combinations_with_replacement(parties,2):
        selection = affinity_table.loc[(all_ids,pair[0]),(all_ids,pair[1])].values
        summary = summary_func(selection)
        affinities.loc[pair[0],pair[1]] = summary
        affinities.loc[pair[1],pair[0]] = summary
        
    return affinities

def unique_comparisons(affinities,features = [],self_comparison=False):
    # assume all affinities tables have the same index
    ids = sorted(list(affinities[0].index))
    data = []
    
    if self_comparison:
        iterator = combinations_with_replacement
    else:
        iterator = combinations
    
    for id1,id2 in iterator(ids,2):
        entry = features + [id1,id2] + [affinities[i].loc[id1,id2] for i in range(len(affinities))]
        data.append(tuple(entry))
    return data



######################################################
## Helpers for Upload_embeddings_to_sql.py
######################################################

agreement_inverses = {'agreement':lambda x: 1-x,'affinity':lambda x: 1.0/x}

dem_color = 'blue'
rep_color = 'red'
party_colors = {'democrat':dem_color,'democrat': dem_color,'republican':rep_color}

punct = re.compile(r"[-\'\. ]")


def normalize_party(party,punct=punct):
    return re.sub(punct,'',party).lower()


def get_nodes(con,time_var,location,time=None,start=None,end=None):
    if start is None:
        command = """select * from nodes_{} where location='{}' and time={}""".format(time_var,location,time)
    else:
        command = """select * from nodes_{} where location='{}' and time>={} and time<={}""".format(time_var,location,start,end)
    nodes = pd.read_sql_query(command,con)
    nodes.set_index('id',inplace=True,drop=False)
    
    return nodes


def get_edges(con,time_var,time,location,start=None,end=None,threshold=0.0):
    if not start:
        command = """select * from affinities_{} where location='{}' and time={}""".format(time_var,location,time)
    else:
        command = """select * from afinities_{} where location='{}' and time>={} and time<={}""".format(time_var,location,start,end)
    edges = pd.read_sql_query(command,con)
    edges.set_index(['id1','id2'],inplace=True,drop=False)
    edges = edges[edges['agreement'] > threshold]
    
    return edges


def get_distances(con,agreement_metric,time_var,location,time=None,transform=None):
    command = """
    select id1, id2, {} from affinities_{} where location='{}' and time={}""".format(agreement_metric,time_var,location,time)
    edges = pd.read_sql_query(command,con)
    if edges.shape[0] == 0:
        print("no samples for {} {}".format(time_var,time))
        return None
    #edges.index = edges.index.values+1
    #edges.loc[0,:] = [edges['id1'][1],edges['id1'][1],np.nan]
    edges.set_index(['id1','id2'],inplace=True)
    edges = edges.loc[:,agreement_metric]
    edges2 = edges.copy()
    edges2.index = pd.MultiIndex.from_tuples([tuple(reversed(pair)) for pair in edges.index],names=('id1','id2'))
    edges = pd.concat((edges,edges2))
    
    sims = edges.unstack('id2')
    sims.sort_index(axis=0,inplace=True)
    sims.sort_index(axis=1,inplace=True)
    
    # sims become distances with the inverse function for embedding
    sims = agreement_inverses[agreement_metric](sims)
    sims.fillna(0,inplace=True)
    
    if transform:
        return transform(sims)
    else:
        return sims


def compute_embedding(sims,senators,last_ids=None,last_embedding=None,last_centers=None):
    parties = senators.party[sims.index]
    parties = parties.apply(normalize_party)
    
    mds = manifold.MDS(n_components=2, max_iter=50000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=-2,metric=True)
    
    if last_embedding is None:
        embedding = mds.fit(sims.values).embedding_
    else:
        # the complicated case of initializing with the last time slice's embedding
        init = np.zeros((sims.shape[0],2))
        locs = sims.index.get_indexer(last_ids)
        keeplocs = np.where(locs >= 0)
        locs = locs[keeplocs]
        init[locs,:] = last_embedding[keeplocs,:]
        newlocs = np.ones(init.shape[0],dtype=bool)
        newlocs[locs] = False
        dem_locs = parties.values=='democrat'
        rep_locs = parties.values=='republican'
        # spread the newly entering politicians around the mean of their party
        # with a variance of 0.2
        indices = np.logical_and(rep_locs,newlocs)
        init[indices,:] = np.random.randn(indices.sum(),2)*0.2 #+ last_centers[0]
        indices = np.logical_and(dem_locs,newlocs)
        init[indices,:] = np.random.randn(indices.sum(),2)*0.2 #+ last_centers[1]
        
        embedding = mds.fit(sims.values,init = init).embedding_
            
    
    center = embedding.mean(axis=0)
    embedding = embedding - center
    #plt.plot(*center,'go')
    
    # now everything might be a little off center if we want the party means to line up on a line
    # through the origin; let's fix that
    rep_center = embedding[parties.values=='republican',:].mean(axis=0)
    dem_center = embedding[parties.values=='democrat',:].mean(axis=0)
    #plt.plot(*rep_center,'go')
    #plt.plot(*dem_center,'go')
    
    # minimize the distance from the origin along a line joining these
    #norm1 = np.sum(rep_center**2)
    #norm2 = np.sum(dem_center**2)
    #dot = np.sum(rep_center*dem_center)
    #delta = (1-norm2/dot)/(norm2/norm1-1)
    #delta = (dot + norm2)/(norm1 + norm2 + 2.0*dot)
    #center = delta*rep_center + (1-delta)*dem_center
    #print(delta)
    #print(center)
    #embedding = embedding - center
    #rep_center = rep_center - center
    #dem_center = dem_center - center
        
    # now rotate to put parties in thier left-right orientation
    norm1 = np.sqrt(np.sum(rep_center**2))
    cos = rep_center[0]/norm1
    sin = rep_center[1]/norm1
    rot = np.array([[cos,-1.0*sin],[sin,cos]])
    embedding = np.apply_along_axis(lambda row: np.dot(row,rot),1,embedding)
    rep_center = np.dot(rep_center,rot)
    dem_center = np.dot(dem_center,rot)
    
    return embedding, sims.index, (rep_center, dem_center)


def plot_network(sims,embedding,senators,xlim=(-0.6,0.6),ylim=(-0.3,0.3),threshold=0.1,location=None,month=None,year=None):
    parties = senators.party[sims.index]
    parties = parties.apply(normalize_party)
    #print(parties[0:10])
    colors = parties.apply(lambda party: party_colors.get(party,'y'))
    
    edges = [[embedding[i, :], embedding[j, :], sims.iloc[i,j],(sims.index[i],sims.columns[j])] for i,j in combinations(np.arange(sims.shape[0]),2) if sims.iloc[i,j] < threshold]
    #print("{} edges".format(len(edges)))
    
    sizes = 200*np.square(1-sims).apply(np.mean,axis=1)
    
    fig = plt.figure(1)
    ax = plt.axes()
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    plt.text(0.9*xlim[0]+0.1*xlim[1],0.9*ylim[1]+0.1*ylim[0],"{}, {}/{}".format(location,month,year), fontsize=18)
    
    if(len(edges) > 0):
        linewidths = [edge[2] for edge in edges]
        edges = LineCollection([edge[0:2] for edge in edges],cmap='Purples')
        edges.set_linewidths(linewidths)
        ax.add_collection(edges)
    
    ax.scatter(embedding[:,0],embedding[:,1],s=sizes,c = colors,marker='o')


def upload_embedding(con,ids,embedding,time_var,time,location):
    nodes = get_nodes(con=con,time_var=time_var,time=time,location=location)
    nodes.loc[ids,['x','y']] = embedding
    
    command = "insert or replace into nodes_{} values "
    command = command.format(time_var) + "{}"
    
    # executemany() should work in place of a loop here but it didn't
    for i in range(nodes.shape[0]):
        con.execute(command.format(tuple(nodes.iloc[i,:])))
        
    con.commit()
    
    # test
    #nodes = con.execute("select * from nodes_{} where time={}".format(time_var,time)).fetchall()
    #print("This is what is in the db after commit:")
    #print(nodes)
    
    
######################################################
## Helpers for Output_json.py
######################################################

def numpy_to_py(x):
    if type(x) is str:
        return x
    else:
        try:
            is_int = (int(x)==x)
        except:
            return x
        else:
            if is_int:
                return int(x)
            else:
                return x
    

def get_party_affinities(networks_con,time_var,start,end,location,agreement_metric='agreement'):
    command = "select party1, party2, time, {} from party_affinities_{} where time >= {} and time <= {} and location = '{}'"
    command = command.format(agreement_metric, time_var, start, end, location)
    
    df = pd.read_sql_query(command, networks_con)
    df.set_index(['party1','party2','time'],inplace=True)
    df = df.unstack()
    df.columns = df.columns.droplevel(0)
    df.sort_index(axis=1,inplace=True)
    
    return df

def write_json(networks_con,congress_con,output_dir,time_var,start,end,location,agreement_threshold=0.25, file_type = 'js'):
    senators = pd.read_sql_query('select * from senators',congress_con)
    senators.set_index('id',inplace=True)
    
    all_nodes = {}
    all_edges = {}
    
    for time in range(start,end+1):
        print("Gathering data for {} {}".format(time_var,time))
        nodes = get_nodes(con=networks_con,time_var=time_var,time=time,location=location)
        nodes.drop(['time','location'],axis=1,inplace=True)
        
        edges = get_edges(con=networks_con,time_var=time_var,time=time,location=location,threshold=agreement_threshold)
        edges.drop(['time','location'],axis=1,inplace=True)
        all_nodes[str(time)] = {i:list(map(numpy_to_py,tuple(nodes.loc[i,:]))) for i in nodes.index}
        all_edges[str(time)] = [list(map(numpy_to_py,tuple(edges.loc[i,:]))) for i in edges.index]
        
    node_keys = dict(zip(nodes.columns,range(len(nodes.columns))))
    edge_keys = dict(zip(nodes.columns,range(len(nodes.columns))))
    
    
    party_agreement = get_party_affinities(networks_con=networks_con,
                                            agreement_metric='agreement',time_var=time_var,
                                            start=start,end=end,location=location)
    party_affinity =  get_party_affinities(networks_con=networks_con,
                                            agreement_metric='affinity',time_var=time_var,
                                            start=start,end=end,location=location)
    
    command = "select time, rate from bill_passage_{} where location='{}' and time >= {} and time <= {}".format(time_var,location,start,end)
    bill_passage = networks_con.execute(command).fetchall()
    bill_passage = dict(bill_passage)
    
    periods = {'year':1,'quarter':4,'month':12}
    period = periods[time_var]
    
    # surely all politicians are between 18 and 110 years old; Strom Thurmond made it to 100 in office
    senators = senators[senators['birth_'+time_var] >= start - period*110]
    senators = senators[senators['birth_'+time_var] <= end - period*18]
    
    # these fail because of the multi-index: json doesn't like strings like "["Dem","Dem"]" which look too much like
    # they're trying to define an Array:
    #party_agreement.to_json(os.path.join(output_dir,'party_agreement.json'),orient='index')
    #party_affinity.to_json(os.path.join(output_dir,'party_affinity.json'),orient='index')
    
    # so we do this:
    party_agreement = {' '.join(i):dict(zip(party_agreement.loc[i,:].index.astype('str'),party_agreement.loc[i,:].values)) for i in party_agreement.index}
    party_affinity = {' '.join(i):dict(zip(party_affinity.loc[i,:].index.astype('str'),party_affinity.loc[i,:].values)) for i in party_affinity.index}
    
    senators = {str(i):dict(zip(senators.loc[i,:].index.astype('str'),map(numpy_to_py,senators.loc[i,:].values))) for i in senators.index}
    
    if file_type.strip('.') in ['js','javascript']:
        with open(os.path.join(output_dir,'party_agreement.js'),'w') as outfile:
            outfile.write('var party_agreement = ')
            outfile.write(json.dumps(party_agreement) + ';')
        with open(os.path.join(output_dir,'party_affinity.js'),'w') as outfile:
            outfile.write('var party_affinity = ')
            outfile.write(json.dumps(party_affinity) + ';')
        with open(os.path.join(output_dir,'bill_passage.js'),'w') as outfile:
            outfile.write('var bill_passage = ')
            outfile.write(json.dumps(bill_passage) + ';')
        with open(os.path.join(output_dir,'node_keys.js'),'w') as outfile:
            outfile.write('var node_keys = ')
            outfile.write(json.dumps(node_keys) + ';')
        with open(os.path.join(output_dir,'edge_keys.js'),'w') as outfile:
            outfile.write('var edge_keys = ')
            outfile.write(json.dumps(edge_keys) + ';')
        
        with open(os.path.join(output_dir,'nodes.js'),'w') as outfile:
            outfile.write('var nodes = ')
            outfile.write(json.dumps(all_nodes) + ';')
        with open(os.path.join(output_dir,'edges.js'),'w') as outfile:
            outfile.write('var edges = ')
            outfile.write(json.dumps(all_edges) + ';')
        with open(os.path.join(output_dir,'senators.js'),'w') as outfile:
            outfile.write('var senators = ')
            outfile.write(json.dumps(senators) + ';')
    
    elif file_type.strip('.') == 'json':
        with open(os.path.join(output_dir,'party_agreement.json'),'w') as outfile:
            json.dump(party_affinity,outfile)
        with open(os.path.join(output_dir,'party_affinity.json'),'w') as outfile:
            json.dump(party_affinity,outfile)
        with open(os.path.join(output_dir,'bill_passage.json'),'w') as outfile:
            json.dump(bill_passage,outfile)
        with open(os.path.join(output_dir,'node_keys.json'),'w') as outfile:
            json.dump(node_keys,outfile)
        with open(os.path.join(output_dir,'edge_keys.json'),'w') as outfile:
            json.dump(edge_keys,outfile)
        
        with open(os.path.join(output_dir,'nodes.json'),'w') as outfile:
            json.dump(all_nodes,outfile)
        with open(os.path.join(output_dir,'edges.json'),'w') as outfile:
            json.dump(all_edges,outfile)
        with open(os.path.join(output_dir,'senators.json'),'w') as outfile:
            json.dump(senators,outfile)
            
    else:
        raise ValueError("invalid file_type arg: {}".format(file_type))

