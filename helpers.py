import pandas as pd 
import numpy as np  
from itertools import tee
import datetime  

def get_top_tasks(tl, N): 
    return sorted(tl, key=lambda x: x['experimental_priority'], reverse=True)[:N] 

def preserve_order_unique_sites(sites):
    unique_sites = []
    for site in sites:
        if site not in unique_sites:
            unique_sites.append(site)
    return unique_sites


def sort_route(route, order):
    return sorted(route, key=lambda x: order.index(x['site_code']))

def sort_by_task_group(lst):
    # Create a dictionary that maps each site_code value to its position in the list
    site_code_positions = {d['site_code']: i for i, d in enumerate(lst)}

    # Get a list of unique site_code values
    site_codes = list(set(d['site_code'] for d in lst))

    # Sort the list of site_code values by their position in the original list
    site_codes.sort(key=lambda x: site_code_positions[x])

    # Initialize the sorted list
    sorted_list = []

    # Iterate over the site_code values
    for site_code in site_codes:
        # Get a sublist of dictionaries with the current site_code value
        sublist = [d for d in lst if d['site_code'] == site_code]
        # Sort the sublist alphabetically by the task group key
        sublist.sort(key=lambda d: d['task group'])
        # Append the sorted sublist to the sorted list
        sorted_list.extend(sublist)

    # Return the sorted list
    return sorted_list


def remaining_weekdays():
    today = datetime.datetime.today()
    day_of_week = today.weekday()
    remaining_days = [datetime.datetime.strftime(today + datetime.timedelta(days=i), '%Y-%m-%d') for i in range(6-day_of_week) if i>0]
    return remaining_days

def create_new_fblocks(sites: list): 

    return [{'task group': f'Facility Block Time -- {site}',
 'number of tasks': 1,
 'total time to complete': 60,
 'experimental_priority': 50,
 'site_code': site,
 'visited': False} for site in sites]
    
    
def parse_redline_links(tg, site, tasks_raw, assignee, unit_num=False): 
    
    """
    tg: task type / group (not including rd) 
    site: site code 
    tasks_raw: dataframe of raw tasks  
    
    outputs markdown list with unit number, rd link 
    """ 
    if unit_num:   
        rows = tasks_raw[(tasks_raw['task_group'] == tg) 
              & (tasks_raw['site_code'] == site) & (tasks_raw['assignee'] == assignee) & (tasks_raw['unit_number'] == unit_num)
                    ][['unit_number', 'link']].sort_values('unit_number')
    else:
        rows = tasks_raw[(tasks_raw['task_group'] == tg) 
              & (tasks_raw['site_code'] == site) & (tasks_raw['assignee'] == assignee)
                    ][['unit_number', 'link']].sort_values('unit_number')
    
    if not rows.empty:
        s = ""
        for unit_num, link in rows[['unit_number', 'link']].values:  
            
            s += f"""<a href="{link}" target = "_blank">{unit_num}</a>"""
            
            #s += f"[unit: {unit_num}]({link})" 
            s += '\t' 

        return s

def format_path(p): 
    
    i = 1
    s = []
    for stop in p: 
        s.append(f"{i}. {stop}\n") 
        i += 1
    
    return ''.join(s)  

# should just be calculating this in the algo but will leave for now 
def manual_time_on_road_pct(ttimes_list, wkday=8, time_buffer=20):  
    
    drive_time = (sum([x - time_buffer for x in ttimes_list if x > 0][:-1]) + ttimes_list[-1]) / 60
    
    return {'pct': round(drive_time / wkday, 2), 'drive_time': round(drive_time, 2)} 

def create_stop_list(sites): 
    s = 1
    current_stop = sites.values[0] 
    stop_num = []
    for stop in sites.values: 
        if stop != current_stop: 
            s += 1  

        current_stop = stop

        stop_num.append(s) 
        
    return stop_num

def create_route_table(route_full): 
    
    d = (
        pd.DataFrame(route_full) 
        .rename(columns={'experimental_priority': 'points'}) 
        .drop('visited', axis=1)
    )  
    
    d['points'] = d['points'].apply(lambda x: str(int(x))) 
    d['total time to complete'] = d['total time to complete'].apply(lambda x: f"{str(int(x))} min") 
    d['stop'] = create_stop_list(d['site_code'])
    
    return d.set_index('stop')

def create_facility_dict(route_full): 
    
    rdf = (
        pd.DataFrame(route_full)
        .groupby('site_code')
        .aggregate({
            'task group': list, 
            'experimental_priority': sum, 
            'total time to complete': sum, 
            'number of tasks': sum
        })
        .assign(total_points = lambda x: x['experimental_priority'].astype(int)) 
        .drop('experimental_priority', axis=1)
    )
    
    rdict = {}
    for site, vals in rdf.iterrows(): 
        rdict[site] = dict(vals) 
        
    return rdict 


def create_site_level_path(route_sites, fd): 
    
    """
    route sites: path of route sites 
    fd: facility dict (like above)
    """
    
    site_level_path = [] 
    visited = []
    for s in route_sites: 
        if s not in visited:  
            d = fd[s] 
            d.update({'site': s})
            site_level_path.append(d) 

            visited.append(s) 
            
    return site_level_path 


def format_site_level_path(site_level_path): 
    
    md_list = []
    
    i = 1
    for stop in site_level_path:  
        md = "" 
        md += f"{i}. **{stop['site']}** *-- {int(stop['total time to complete'])} minutes of tasks -- {stop['total_points']} points collected*\n"   
        for tg in stop['task group']: 
            md += f"\t- {tg}\n" 
        
        md += "  \n" 
        md_list.append(md)

        i += 1

    return md_list 


def get_task_notes(task_group_route, full_tasks, name, note_tasks=['In House Maintenance', 'Trash Hauling', 'Gate Service Requested']): 
    
    """
    basic idea is that the following task groups need more info provided on them 
    
    - in house maintenance 
    - trash hauling 
    
    if these are in the optimal route, construct a table with notes for fs to see 
    
    TODO: match misnames 
    """ 
    
    tables = {}
    for task in note_tasks:  
        
        sites = [x.split('--')[-1].strip() for x in task_group_route if task in x] 
        if len(sites) > 0: 
            fdf = full_tasks[(full_tasks['task_group'] == task) & (full_tasks['site_code'].isin(sites)) & (full_tasks['assignee'] == name)][['site_code', 'task_group', 'unit_number', 'task_description', 'created_by']].set_index('site_code').sort_index()
            
            tables[task] = fdf 
            
    return tables

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def travel_times_markdown(route_sites, travel_times):  
    mroute = ['Home'] + route_sites + ['Home']  
    
    md = ""
    i = 0
    for a,b in pairwise(mroute):  
        if a != b:
            md += f"{a} $\longrightarrow$ {b} ~ {round(travel_times[i])} minutes\n\n"
        i += 1  
        
    return md