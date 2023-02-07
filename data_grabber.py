import boto3 
import json  
from io import StringIO 
import pandas as pd    
import os 
import pickle  

import datetime 
from collections import Counter

def grab(MASTER_ACCESS_KEY, MASTER_SECRET):  
    
#    def remaining_weekdays():
#        today = datetime.datetime.today()
#        day_of_week = today.weekday()
#        remaining_days = [datetime.datetime.strftime(today + datetime.timedelta(days=i), '%Y-%m-%d') for i in range(6-day_of_week) if i>0]
#        return remaining_days
#
#    def get_pto(s3):  
#        """
#        returns a dict of name: pto days this week 
#        """
#        pto_files = s3.list_objects(Bucket='fs-optimization', Prefix='routes/pto/') 
#        pto_fpaths = [x['Key'] for x in pto_files['Contents'] if '202' in x['Key']]   
#
#        # filter to only days this week 
#        rweekdays = remaining_weekdays()  
#        ndays = len(rweekdays)
#
#        # count this weeks pto by fs 
#        fs_count = []
#        for p in pto_fpaths: 
#            fs = p.split('/')[-1].replace('.json', '')
#            p = p.split('/')[-2] 
#            if p in rweekdays: 
#                fs_count.append(fs) 
#
#        fs_pto_counts = dict(Counter(fs_count))
#
#        return fs_pto_counts

    
    # --- s3 client --- 
    s3 = boto3.client('s3', region_name = 'us-west-1', 
          aws_access_key_id=MASTER_ACCESS_KEY, 
          aws_secret_access_key=MASTER_SECRET) 
    
    def read_data(f, bucket='fs-optimization', idx_col=None):
        data = s3.get_object(Bucket=bucket, Key=f)['Body'].read().decode('utf-8') 
        if idx_col is None:
            data = pd.read_csv(StringIO(data)) 
        else:
            data = pd.read_csv(StringIO(data), index_col=idx_col)

        return data 
    
    # --- read pkl files --- 
    
    fs_homes = pickle.loads(s3.get_object(Bucket='fs-optimization', 
                                                    Key='fs-info/fs_homes_dict.pkl')['Body'].read())  
    
    fs_routes = pickle.loads(s3.get_object(Bucket='fs-optimization', 
                                                    Key='tsp/fs_routes.pkl')['Body'].read())  
    
    update_file = json.loads(s3.get_object(Bucket='fs-optimization', 
                                           Key='routes/last-update/last_update.json')['Body'].read()) 
    
    last_file_key = update_file['last_update']  
    
    tasks_raw_update_file_key = json.loads(s3.get_object(Bucket='fs-optimization', Key='task-priorities/last-update/last_update.json')['Body'].read())['last_update']  
    print(f'task-priorities/{tasks_raw_update_file_key}.csv')
    tasks_raw = read_data(f'task-priorities/{tasks_raw_update_file_key}.csv')  
    tg = read_data(f'task-priorities/task-groups/{tasks_raw_update_file_key}.csv')
    
    task_routes = pickle.loads(s3.get_object(Bucket='fs-optimization', 
                                                    Key=f'routes/{last_file_key}.pkl')['Body'].read())  
    
    fs_homes_new = read_data(f='fs-info/fs_home_locations_revised.csv') 
    
    # bring in dtable 
    dtable_update_file = pickle.loads(s3.get_object(Bucket='fs-optimization', 
                                           Key='distance-matrices/last-update/update.pkl')['Body'].read()) 
    
    lfk = dtable_update_file.get('last-update') 
    assert lfk is not None  
    print(f"dtable last updated {lfk}")
    dtable = read_data(f=f'distance-matrices/{lfk}.csv') 
    
    # fs workday data 
    fs_workdays = read_data(f="fs-info/fs_workday_data.csv", idx_col=0)  
    
    # pto counts 
    #pto_counts = get_pto(s3) 
    #print(pto_counts)
    
    
    #### new route query ### 
    
    """
    1. pull last route pull and store date 
    2. look at revisions/ folder for any revisions for that date 
    3. look for any pto 
    3. update the routes dict with any revisions made 
    4. in app.py, note if the route was revised 
    """
    
    def pull_revisions(last_file_key, routes, verbose=True):  
        
        if verbose: 
            print('looking for revisions...')
        
        contents = s3.list_objects(Bucket='fs-optimization', Prefix='routes/modifications/') 

        revised_fpaths = [x['Key'] for x in contents['Contents'] if last_file_key in x['Key']] 

        if len(revised_fpaths) > 0:  
            
            if verbose: 
                print('revisions found!')
            
            for f in revised_fpaths:  
                name = f.split('/')[-1].strip('.pkl').split('_')[-1]  
                # pull revised route
                r = pickle.loads(s3.get_object(Bucket='fs-optimization', Key=f)['Body'].read()) 
                # update routes dict 
                routes.update(r)

            return 1

        return 0 
    
    # this will make *in-place* changes to task routes dict 
    # a revised route will contain the key 'reoptimized_route' 
    pull_revisions(last_file_key, task_routes) 
    
    
    fblocks = pickle.loads(s3.get_object(Bucket='fs-optimization', 
                                                    Key=f'routes/facility-blocks/{last_file_key}.pkl')['Body'].read()) 
    
    # --- coverage --- 
    def read_coverage_file(s3, fpath): 
        f = pickle.loads(s3.get_object(Bucket='fs-optimization', Key=fpath)['Body'].read())  
        return f

    def get_coverage_files(s3, dt): 
        res = s3.list_objects(Bucket='fs-optimization', Prefix=f'routes/coverage/{dt}/')  

        if 'Contents' in res: 
            conts = {}
            for obj in res['Contents']: 
                # conts[fs] = cov dict obj  
                name = obj['Key'].split('/')[-1].strip('.pkl') 
                cov_file = read_coverage_file(s3, obj['Key'])

                conts[name] = cov_file 

            return conts 
        else: 
            return {} 
        
    
    cov = get_coverage_files(s3, last_file_key)
    
    
    return {'homes': fs_homes, 'routes': fs_routes, 
            'opt_routes': task_routes, 
            'fs_homes_new': fs_homes_new, 
            'tasks_raw': tasks_raw, 
            'last_update': last_file_key, 
            'task_groups': tg, 
            'dtable': dtable, 
            'routes_path': f'routes/{last_file_key}.pkl', # in case an re-optimization is made 
            'fs_workdays': fs_workdays, 
            'fblocks': fblocks, 
            'coverage': cov
            #'pto_counts': pto_counts
           } 



def upload_file_to_s3(data, bucket, path, fname, MASTER_ACCESS_KEY, MASTER_SECRET, file_type='csv'): 
    
    s3 = boto3.client('s3', region_name = 'us-west-1', 
    aws_access_key_id=MASTER_ACCESS_KEY, 
    aws_secret_access_key=MASTER_SECRET) 
    
    try:  
        if file_type == 'csv':

            s3.upload_fileobj(BytesIO(bytes(data.to_csv(line_terminator='\r\n', index=False), encoding='utf-8')), Bucket=bucket, Key=f'{path}{fname}.csv')   

            print(f'successful upload! {fname} uploaded to {bucket}/{path}') 
            return 1  
        
        elif file_type == 'json':   
            
            s3.put_object(Body=(bytes(json.dumps(data).encode('UTF-8'))), Bucket=bucket, Key=f'{path}{fname}.json')  
            
            print(f'successful upload! {fname} uploaded to {bucket}/{path}') 
            return 1   
        
        elif file_type == 'pkl': 
            
            s3.put_object(Body=pickle.dumps(data), Bucket=bucket, Key=f'{path}{fname}.pkl')
            
            print(f'successful upload! {fname} uploaded to {bucket}/{path}') 
            return 1
            
        else: 
            print('cannot upload file type yet...') 
            return 0  
    
    except Exception as e: 
        
        print(e) 
        return 0 