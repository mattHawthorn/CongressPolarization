#encoding:utf-8

import os
import sys
from helpers import *

# initialize the tables
os.chdir('..')

# initialize the database
input_db = sys.argv[1]#'congress.db'
output_db = sys.argv[2]#'congress_networks.db'
output_db_name = os.path.splitext(output_db)[0]
schema = read_schema(os.path.join('config',output_db_name + '_schema'),extensions=['.yml','.json'],legal_types=sqlite_types)

os.chdir('data')
init_sqlite_db(db_path=output_db,schema=schema,init_index=False)

# connect
con = sqlite3.connect(input_db)
cur = con.cursor()

out_con = sqlite3.connect(output_db)
out_cur = out_con.cursor()


# parameters
time_cols = sys.argv[3::2]
lag_list = sys.argv[4::2]
lag_list = [int(lag) for lag in lag_list]
default_age = 50
periods = dict(zip(['year','quarter','month'],[1,4,12]))
lags = dict(zip(time_cols,lag_list))
# don't have population data earlier than this
min_pop_year = 1905
state_stat = 'Density Rank'

min_bills = 5
min_voters = 10


# External data to join to nodes for the temporally dependent variables
senator_birthdays = pd.read_sql_query("select id,state,"+','.join(['birth_'+unit for unit in time_cols])+" from senators",con)
senator_birthdays.set_index('id',inplace=True)
senator_birthdays.columns = pd.Index(['state']+time_cols)

state_data = pd.read_csv('state_population.csv')
state_data = state_data.set_index(['state','stat'])
state_data.columns = pd.Index(state_data.columns,dtype='int')
# this ends up as a Series with a MultiIndex
state_data = state_data.stack()

print("Extracting all agreement networks for time variables {} with lags {}".format(time_cols,lag_list))

# Compute and upload the network data
for time_col in time_cols:
    affinity_fields = schema['affinities_'+time_col]['fields']
    node_fields = schema['nodes_'+time_col]['fields']
    party_affinity_fields = schema['party_affinities_'+time_col]['fields']
    senator_experience = {}
    senator_ages = {}
    per_year = periods[time_col]
    lag = lags[time_col]
    
    locations = cur.execute("select distinct location from bills").fetchall()
    locations = list(zip(*locations))[0]
    
    for location in locations:
        print("Computing agreements for {} by {} with lag {}".format(location,time_col,lag))
        print()
        
        time_vals = cur.execute("select distinct {} from votes where location = '{}'".format(time_col,location)).fetchall()
        time_vals = list(zip(*time_vals))[0]
        
        # time_vals may have gaps so we loop over the min,max range
        for time_val in range(min(time_vals),max(time_vals)+1):
            votes = vote_table(con,time_col,time_val,location,lag=lag)
            senators = [i[0] for i in votes.index]
            
            for key in senator_ages:
                senator_ages[key] = senator_ages[key]+1
            
            # turning month/quarter 0 values into their more common representation
            y1 = (time_val-lag) // per_year
            r1 = (time_val-lag) % per_year
            y2 = time_val // per_year
            r2 = time_val % per_year
            
            if r1 == 0:
                r1 = per_year
                y1 -= 1
            if r2 == 0:
                r2 = per_year
                y2 -= 1
            
            print("{}: {}/{}-{}/{} voters: {} bills: {}".format(time_col,r1,y1,r2,y2,votes.shape[0],votes.shape[1]))
                
            # build up data for the nodes 1 senator at a time
            node_data = []
            
            for senator in senators:
                try:
                    birth = senator_birthdays.loc[senator,time_col]
                except:
                    birth = None
                
                if pd.isnull(birth) or birth is None:
                    try:
                        age = int(senator_ages[senator]//per_year)
                    except:
                        age = default_age
                        senator_ages[senator] = age*per_year
                else:
                    # age as an integer
                    age = int((time_val - birth)//per_year)
                

                if senator in senator_experience:
                    senator_experience[senator] = senator_experience[senator] + 1
                else:
                    senator_experience[senator] = 1
                # experience as a floating point number of years
                experience = senator_experience[senator]/per_year
                
                year = time_val/per_year
                if year > min_pop_year:
                    if year <= 1905:
                        year = 1910
                    elif year >=2015:
                        year = 2010
                    round_year = round(year,-1)
                    state = senator_birthdays.loc[senator,'state']
                    try:
                        pop_density = state_data[(state,state_stat,round_year)]
                    except:
                        pop_density = None
                else:
                    pop_density = None
                    
                # No x and y data yet
                node_data.append((location,time_val,senator,age,experience,pop_density,None,None))
            
            # discard sparse samples; have to run above code to be sure that time-dependent node variables
            # get incremented
            if votes.shape[1] < min_bills:
                print("Fewer than {} bills; skipping.".format(min_bills))
                continue
            if votes.shape[0] < min_voters:
                print("Fewer than {} voters; skipping.".format(min_voters))
                continue
            
            affinities = affinity_table(votes,affinity_func=voter_perplexity,transform=np.exp)
            agreements = affinity_table(votes,affinity_func=voter_agreement_pos)
            
            party_affinities = interparty_affinity(affinities,summary_func=lambda x: np.exp(np.mean(np.log(x))))
            party_agreements = interparty_affinity(agreements,summary_func=np.mean)
            
            command = "select aye,nay from bills where {}<={} and {}>={} and location='{}'".format(time_col,time_val,
                                                                                                 time_col,time_val-lag,location)
            df = pd.read_sql(command,con)
            if df.shape[0] == 0:
                rate_of_passage = 0
            else:
                rate_of_passage = (df['aye'] > df['nay']).sum()/df.shape[0]
                print(rate_of_passage)
            
            affinities.columns = affinities.columns.droplevel(1)
            affinities.index = affinities.index.droplevel(1)
            agreements.columns = agreements.columns.droplevel(1)
            agreements.index = agreements.index.droplevel(1)
            
            voter_data = unique_comparisons([agreements,affinities],features=[location,time_val])
            party_data = unique_comparisons([party_agreements,party_affinities],features=[location,time_val],
                                           self_comparison=True)
            
            
            
            insert_rows(out_cur, 'affinities_'+time_col, voter_data,affinity_fields)
            insert_rows(out_cur, 'party_affinities_'+time_col, party_data, party_affinity_fields)
            insert_rows(out_cur, 'nodes_'+time_col, node_data, node_fields)
            
            command = 'insert or replace into bill_passage_{} values {}'.format(time_col,(location,time_val,rate_of_passage))
            out_cur.execute(command)
            
            out_con.commit()
            
    print()


print("all networks extracted for time variables {} with lags {}".format(time_cols,lag_list))
print("run 'Create_indices.py {}' to make indices for fast queries".format(output_db))

# close the connection
cur.close()
con.close()
out_cur.close()
out_con.close()

exit()

