import datetime 

def remaining_weekdays():
    today = datetime.datetime.today()
    day_of_week = today.weekday()
    remaining_days = [datetime.datetime.strftime(today + datetime.timedelta(days=i), '%Y-%m-%d') for i in range(6-day_of_week) if i>0]
    return remaining_days

def filter_task_list(name, task_list, dtable, max_dist=3*60*60): 
    """
    from chatgpt 
    
    filters a task list down to exclude tasks at facilities over a certain time threshold away 
    
    basically accounting for process errors
    """
    
    filtered_tasks = []
    excluded_tasks = []
    for task in task_list:
        start = name
        end = task['site_code']
        mask = (dtable['start'] == start) & (dtable['end'] == end) & (dtable['value'] <= max_dist)
        if mask.any():
            filtered_tasks.append(task)
        else:
            excluded_tasks.append(task)
    return filtered_tasks, excluded_tasks


def prepare_data(red_name, 
                 name,  
                 task_groups,
                 dtable,
                 misnamed=['Malik  Goller', 'Jack  Mabrey', 'Michael  Richardson', 'Wesley  Applebury'], 
                 return_missing=False):  
    
    if red_name in misnamed: 
        print("misnamed FS")
        name = red_name.replace('  ', ' ')
    
    fs_df = task_groups.query(f"FS == '{red_name}'").reset_index(drop=True)
    task_order = list(fs_df[['task group', 'total time to complete', 'experimental_priority', 'site_code']].iterrows())  

    task_list = []
    for idx, r in fs_df[['task group', 'number of tasks','total time to complete', 'experimental_priority', 'site_code']].iterrows(): 
        r = dict(r) 
        r['visited'] = False 
        task_list.append(r) 
        
    fs_locs_in_dtable = dtable[dtable['start'] == name]['end'].unique().tolist() 
    task_list_revised = [x for x in task_list if x['site_code'] in fs_locs_in_dtable]  
    
    if return_missing: 
        missing = [x for x in task_list if x['site_code'] not in fs_locs_in_dtable]  
        return {'name': name, 'task_list': task_list_revised, 'missing': missing}
    
    return {'name': name, 'task_list': task_list_revised} 
    
def get_facility_block_time_multiplier(ndays_remaining, BASE_SCORE=50): 
    m = .5 - (ndays_remaining / 10)
    return round(BASE_SCORE * (1+m))

def prepare_data_with_facility_blocks(fblocks, red_name,  
                 name,  
                 task_groups,
                 dtable,
                 misnamed=['Malik  Goller', 'Jack  Mabrey', 'Michael  Richardson', 'Wesley  Applebury'], 
                 return_missing=False):   
    
    fields = ['task group', 'number of tasks', 'total time to complete', 'experimental_priority', 'site_code', 'task_ids'] # 'task_ids' 
    
    ndays_remaining = len(remaining_weekdays()) 
    
    if red_name in misnamed: 
        name = red_name.replace('  ', ' ') 
    
    fs_df = task_groups.query(f"FS == '{red_name}'").reset_index(drop=True) 
    task_order = list(fs_df[fields].iterrows())   

    task_list = []
    for idx, r in fs_df[fields].iterrows(): 
        r = dict(r) 
        r['visited'] = False 
        task_list.append(r) 
        
    fs_locs_in_dtable = dtable[dtable['start'] == name]['end'].unique().tolist() 
    task_list_revised = [x for x in task_list if x['site_code'] in fs_locs_in_dtable]   
    
    fs_sites = set([x['site_code'] for x in task_list_revised]) 
    fs_blocks = [x for x in fblocks if x['site_code'] in fs_sites] 

    new_task_list_with_blocks = task_list_revised + fs_blocks 
    
    # get filtered task list
    new_task_list_with_blocks, excluded_tasks = filter_task_list(name, new_task_list_with_blocks, dtable)
    
    print(f"{name} has {len(excluded_tasks)} excluded tasks")
    
    # scale block points based on days remaining 
    for task in new_task_list_with_blocks: 
        if 'Facility Block Time' in task['task group']: 
            task['experimental_priority'] = get_facility_block_time_multiplier(ndays_remaining) 
            task['task_ids'] = None

    if return_missing: 
        missing = [x for x in task_list if x['site_code'] not in fs_locs_in_dtable]  
        return {'name': name, 'task_list': new_task_list_with_blocks, 'missing': missing, 'excluded': excluded_tasks}
    