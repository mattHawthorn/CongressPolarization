#encoding:utf-8

# structure of govtrack data is this: (lines preceded by '-' are folders)

# -congress term
#     -votes
#          -year (2 possible)
#               -billno (h31 for house, s31 for senate)
#                    data.xml
#                         <roll where="house" session="1" year="1790" roll="36" source="keithpoole" 
#                          datetime="1790-02-12" updated="2015-07-26T10:18:48-04:00" 
#                          aye="43" nay="11" nv="11" present="0">
#
#                         <category>unknown</category>
#
#                         <type>TO REFER TO A COMMITTEE THE MEMORIAL TO THE SENATE AND HOUSE, FOR THE ABOLITION OF SLAVERY.</type>
#
#                         <question>TO REFER TO A COMMITTEE THE MEMORIAL TO THE SENATE AND HOUSE, FOR THE ABOLITION OF SLAVERY.</question>
#
#                         <required>unknown</required>
#
#                         <result>unknown</result>
#
#                         <option key="+">Yea</option>
#                         <option key="-">Nay</option>
#                         <option key="0">Not Voting</option>
#
#                         <voter id="400829" vote="+" value="Yea" state="MA"/>
#                         <voter id="401379" vote="+" value="Yea" state="NY"/>
#                         <voter id="401623" vote="+" value="Yea" state="NJ"/>
#                         <voter id="401881" vote="+" value="Yea" state="VA"/>
#                         <voter id="402174" vote="+" value="Yea" state="NJ"/>
#                         <voter id="402671" vote="+" value="Yea" state="PA"/>
#
#                         ...

import os
import sys
from helpers import *

os.chdir('..')

# initialize the database
output_db = sys.argv[1] #'congress.db'

schema = read_schema(os.path.join('config',os.path.splitext(output_db)[0]+'_schema'),extensions=['.yml','.json'],legal_types=sqlite_types)

os.chdir('data')
init_sqlite_db(db_path=output_db,schema=schema,init_index=False)

# connection to be used throughout this session
con = sqlite3.connect(output_db)
cur = con.cursor()


# perform the import
os.chdir('congress')
vote_fields = schema['votes']['fields']
bill_fields = schema['bills']['fields']

# get the list of subdirectories, one for each session
congresses = sorted([name for name in os.listdir() if re.match(congress_num,name)],key=lambda s: int(s))

for congress in congresses:
    #if int(congress) != 35:
    #    continue
    
    vote_dir = os.path.join(congress,'votes')
    years = sorted(os.listdir(vote_dir))
    
    # previously imported bills:
    imported_bills = cur.execute("select location, roll from bills where session={}".format(congress)).fetchall()
    imported_bills = set(imported_bills)
    
    # for every year of that session
    for year in years:
        bills = os.listdir(os.path.join(vote_dir,year))
        bills = sorted([num for num in bills if re.match(congress_bill,num)], 
                       key = lambda s: int(s[1:]))
        
        bill_data = []
        
        # for every bill in that year
        for bill in bills:
            start_char = bill[0]
            if start_char == 's':
                location = 'senate' 
            elif start_char == 'h':
                location = 'house'
            else:
                continue
            
            if (location,int(bill[1:])) in imported_bills:
                continue
            
            # list the files in the bill dir and parse the xml
            bill_dir = os.path.join(vote_dir,year,bill)
            files = os.listdir(bill_dir)
            xmlfiles = [file for file in files if file.endswith('.xml')]
            
            with open(os.path.join(vote_dir,year,bill,xmlfiles[0]), 'r') as infile:
                # get the xml object representing all the votes
                data = objectify.parse(infile).getroot()
                
            # get the bill data from the xml object
            new_bill = get_bill_data(data)
            
            # append to the list for that year
            bill_data.append(tuple(new_bill))
            
            # get the vote data from the xml object
            voters = data.voter
            vote_data = [tuple(new_bill[0:6] + [voter.attrib.get(attr,None) for attr in vote_fields[6:]]) for voter in voters]
            
            # print a random vote every now and then
            #if np.random.rand() > 0.9997:
            #    print(new_bill)
            #    print(vote_data[np.random.randint(0,len(vote_data))])
            
            # insert the vote data into the db for each bill
            insert_rows(cur,'votes',vote_data,vote_fields,'ignore')
        
        # insert the bill data into the db for each year
        insert_rows(cur,'bills',bill_data,bill_fields,'ignore')
        
        # commit the changes
        con.commit()
        
    print("Congress {} imported".format(congress))


# Now import senator data
os.chdir('..')
os.chdir('congress-legislators')

# read all the senator data
historic_senators = pd.read_csv('legislators-historic.csv')
current_senators = pd.read_csv('legislators-current.csv')

# merge historic and current
current_senators.set_index('govtrack_id',verify_integrity=True,inplace=True)
historic_senators.set_index('govtrack_id',verify_integrity=True,inplace=True)
senators = pd.concat([historic_senators,current_senators],axis=0,join='outer',verify_integrity=True)

# add columns
df = senators.birthday.apply(year_quarter_month)
df.columns = pd.Index(['birth_year','birth_quarter','birth_month'])
senators = pd.concat([senators,df],axis=1,join='outer')
senators['id'] = senators.index

# drop columns
senator_fields = schema['senators']['fields']
senators = senators[senator_fields]

# collect the senator data in a form that sqlite3 likes
senator_types = schema['senators']['types']
data = []
for index in senators.index:
    row = senators.loc[index,]
    row_data = []
    for i in range(len(row)):
        row_data.append(type_or_null(row.iloc[i],typedict[senator_types[i]]))
    data.append(tuple(row_data))

# import the data
insert_rows(cur,'senators',data,senator_fields)
con.commit()


# Tests
# check that all years are present
for time_col in ['year','quarter','month']:
    times = cur.execute("select distinct {} from votes".format(time_col)).fetchall()
    times = np.array(sorted(list(zip(*times))[0]))
    diffs = np.diff(times)
    diffs = diffs !=1
    if np.sum(diffs) != 0:
        print("Warning: gaps are present in the {}s imported to the db".format(time_col))
        print(times[0:-1][diffs])
        print(times[1:][diffs])
        break


# now handled by Create_indices.py
## create indices for fast queries
#for table, table_info in schema.items():
#    if 'indices' in table_info:
#        print("Creating indices for table {}".format(table))
#        create_index(con,**table_info)


# close the connection
cur.close()
con.close()

exit()

