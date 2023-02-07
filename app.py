import streamlit as st 
import folium
import pickle 
import pandas as pd  
import numpy as np    
import datetime

import geocode 
import data_grabber 
from helpers import * 
from passwords import PASSWORDS 

from algorithm import algo, prep_data, params 

param_dict = params.param_dict



MASTER_ACCESS_KEY = st.secrets['MASTER_ACCESS_KEY']  
MASTER_SECRET = st.secrets['MASTER_SECRET'] 

st.set_page_config(layout='centered', page_icon='🚗', page_title="FS Route Planner")  

def blank(): return st.write('') 

@st.cache(allow_output_mutation=True, ttl=.5*60*60, suppress_st_warning=True) 
def grab_data(MASTER_ACCESS_KEY, MASTER_SECRET):  
    
    print('grabbing data') 
    
    data = data_grabber.grab(MASTER_ACCESS_KEY, MASTER_SECRET) 
    
    return data, pd.read_csv('rd-site-locations.csv', index_col=0)


data, sites = grab_data(MASTER_ACCESS_KEY, MASTER_SECRET)      

dat = data['routes'] 
fs_homes = data['homes']  
routes = data['opt_routes'] 
homes = data['fs_homes_new']  
tasks_raw = data['tasks_raw']
last_update = data['last_update']  
task_groups = data['task_groups'] 
dtable = data['dtable']
fs_workdays = data['fs_workdays'] 
fblocks = data['fblocks']  
cov = data['coverage']
pto_counts = data.get('pto_counts') #data['pto_counts']  


n_weekdays = len(remaining_weekdays()) 

st.subheader('FS Route Planner')
st.caption(f"Routes last calculated for: **{last_update}**")

task_tab, route_build_tab, pto_tab, appt_tab = st.tabs(['View Route', 'Re-Optimize', "Submit Off Days", 'Submit Future Appointment'])    

def password_authenticate(fs_password, PASSWORDS):  
    
    if fs_password == PASSWORDS['admin']: 
        return True
    
    if fs_name == 'Michael  Richardson': 
        if PASSWORDS.get(fs_name.replace('  ', ' ')) == fs_password: 
            return True   
        else:
            return False 
    else:
        if PASSWORDS.get(fs_name) == fs_password: 
            return True 
        else: 
            return False 
    
    return False 
    

with task_tab:   
    with st.form(key='form0'): 
        
        fs_name = st.selectbox('Name', sorted(list(routes.keys())))  
        
        fs_password = st.text_input("Password") # type='password' 
        
        search = st.form_submit_button('Get Route')  
        
    if search:  
        
        password_valid = password_authenticate(fs_password, PASSWORDS) 
        
        if password_valid:  
            with st.expander("Task Group Breakdown"): 
                s = '- **Auction Unit**: Auction Unit, Mgr Special Auction Unit  \n- **Break In**: Break In  \n- **CLNRR**: CLNRR  \n- **Customer Appointment (Unit Task)**: Customer Appointment (Unit Task)  \n- **Customer Reported**: Customer Reported  \n- **DOT: Add**: DOT: Add  \n- **Gate Preventative Maintenance**: Gate Preventative Maintenance  \n- **Gate Service Requested**: Gate Service Requested  \n- **Improperly Locked**: IFNL, AFNL, Improperly Locked  \n- **In House Maintenance**: In House Maintenance  \n- **Insurance Claim Inspection**: Insurance Claim Inspection  \n- **Kiosk**: Kiosk  \n- **NROL**: NROL  \n- **Overlock**: Overlock: Reverse, DOT: Reverse, Overlock: Add  \n- **Site Audit Action Item**: Site Audit Action Item  \n- **Trash Hauling**: Trash Hauling  \n- **Unit Check**: Unit Check  \n'
                
                st.caption("This list shows task groups in bold, as well as the individual tasks that make up a task group")
                st.markdown(s)
                
            st.success("Valid Password")   
            
            # coverage = routes[fs_name].get('coverage', False)  
            coverage = fs_name in cov
            if coverage: 
                st.info(f"{fs_name} is covering for {cov[fs_name]['primary_fs']} in this route")   
                primary_fs = cov[fs_name]['primary_fs']
            else: 
                primary_fs = fs_name 
            
                
            plot_route = True 
            res = routes[fs_name]   
            revised = 'reoptimized_route' in res   
            
            
             
            if coverage: 
                home_loc = cov[fs_name]['cover_address'] 
            else:   
                home_loc = fs_homes.get(fs_name.replace('  ', ' '), np.nan) 
                if pd.isnull(home_loc):  
                    st.error(f"Missing a home location for {fs_name}")
#                house = homes.query(f"redline_name == '{fs_name}'")
#                if not house.empty: 
#                    home_loc = tuple(house[['lat', 'lng']].values[0]) 
#                else: 
#                    house = homes.query(f"redline_name == '{fs_name.replace('  ', ' ')}'")  
#                    if not house.empty: 
#                        home_loc = tuple(house[['lat', 'lng']].values[0]) 
#                    else: 
#                        home_loc = np.nan


            fitness = res['fitness']   
    
            sorder = preserve_order_unique_sites(res['route_sites']) 
            sorted_route = sort_route(res['route_full'], sorder)
            #sorted_route = sort_by_task_group(res['route_full'])


            facility_dict = create_facility_dict(sorted_route) 
            site_path = create_site_level_path(res['route_sites'], facility_dict)  
            site_path_markdown = format_site_level_path(site_path)   
            

            # doing this bc algo uses "buffer time" that is probably excessively big  
            tor = manual_time_on_road_pct(fitness['travel_times']) 
            time_on_rd = tor['pct'] 
            drive_time = tor['drive_time']  
            
            appt_notes_present = False 
            if revised:  
                
                try:
                    future_appt = res['revision']['value'].get('submission_date', False)  
                except: 
                    future_appt = False  
                    
                site_specific = res['revision']['value']['site'].startswith('RD')
                
                if future_appt: 
                    st.info(f"Appointment submitted on {res['revision']['value'].get('submission_date', '')}") 
                else:
                    #st.info(f"This route was revised on {res['reoptimized_timestamp'].split('T')[0]}")   
                    st.info(f"Reoptimized route: {res['revision']['type'].replace('_', ' ')}")  
                    
                if not site_specific: 
                    st.info(f"**Non Site-Specific Appointment on Route at {res['revision']['value']['site']['time']}**")
                    
                # also get appt info if it's there  
                # {'revision': {'type': 'appointment', 'value': sched_params['notes']}}
                
                if 'revision' in res: 
                    if res['revision'].get('type', '') == 'appointment': 
                        appt_notes_text = f"**Location**: {res['revision']['value'].get('site', '')}\n\n**Notes**: {res['revision']['value'].get('notes', '')}\n\n **Time**: {res['revision']['value'].get('time', '')}  \n\n**Length**: {res['revision']['value'].get('appt_length', 'appt_length')} minutes\n\nScheduled by: {res['revision']['value'].get('scheduled_by', '')}"  
                        appt_notes_present = True
                

            st.markdown('**Optimal Route**') 
            st.write(f"Route will score {fs_name.split()[0]} **{int(fitness['score'])}** task points in **{round(fitness['cost_less_transition_pen'], 2)}** hours, and include **{drive_time}** hours of driving (**{round(time_on_rd * 100)}%** of the day)") 
            blank()  
                        
            last_site = None 
            i = 0
                
            md_list = []
            for s in sorted_route:  
                md = "\n\n" 
                    
                tg = s['task group'].split('--')[0].strip() 
                site = s['site_code'] 
                    
                if site != last_site:  
                    stop = site_path[i]  
                    md += f"{i+1}. <u>**{site}</u> *-- {int(stop['total time to complete'])} minutes of tasks (+ 10 min site check) -- {stop['total_points']} points collected***\n"  
                    # md += "\t- Site Check -- 0 points -- 10 minutes<br>"
                    last_site = site
                    i += 1
                    
                # check if unit num in task group 
                if s['task group'].count('--') == 2:  
                    unit_num = s['task group'].split('--')[1].strip()  
                    redline_links = parse_redline_links(tg, site, tasks_raw, primary_fs, unit_num=unit_num) #redline links will be in primary fs name 
                else: 
                    redline_links = parse_redline_links(tg, site, tasks_raw, primary_fs)  
                    
                md += f"\t- {s['task group']}: {redline_links} -- *{int(round(s['experimental_priority']))} points* -- *{int(round(s['total time to complete']))} minutes*<br>"
                md_list.append(md) 
                    
            path_str = "".join(md_list)
            st.markdown(path_str, unsafe_allow_html=True) 
            
#            for s in site_path_markdown: 
#                st.markdown(s) 
#                blank() 

            note_tables = get_task_notes(task_group_route=res['route_task_groups'], 
                           full_tasks=tasks_raw, 
                           name=primary_fs) 

            blank()
#
            if appt_notes_present: 
                with st.expander("Appointment Notes"): 
                    st.write(appt_notes_text)
            
            if len(note_tables.keys()) > 0: 
                with st.expander("Supplementary Task Notes"):
                    for task in note_tables:  
                        st.markdown(f"**{task}**")
                        st.write(note_tables[task]) 


            ttimes = travel_times_markdown(res['route_sites'], res['fitness']['travel_times'])  
            with st.expander("Travel Times"): 
                st.info("**Note**: a **20 minute** buffer time was added to all travel times here, **with the exception of the final leg of the day**, to account for parking upon new facility arrival, traffic delays, etc.")
                st.markdown(ttimes) 

            with st.expander("Excluded Task Groups"): 
                visited_sites = set(res['route_sites'])
                unvisited_df = (
                    pd.DataFrame(res['unvisited'])
                    .drop(['visited', 'task_ids'], axis=1)
                    .set_index('task group') 
                    .rename(columns={'experimental_priority': 'points', 'total time to complete': 'time to complete'}) 
                )  
                
                unvisited_df['time to complete'] = unvisited_df['time to complete'].apply(lambda x: str(int(x)) + ' min')
                unvisited_df['points'] = unvisited_df['points'].astype(int)

                for site in visited_sites:
                    st.markdown(f"**{site}**") 
                    site_df = unvisited_df.query(f"site_code == '{site}'")
                    if site_df.empty: 
                        st.write(f"All tasks picked up at {site}") 
                    else:
                        st.write(site_df) 

                other_sites = unvisited_df[~unvisited_df['site_code'].isin(visited_sites)]

                st.markdown("**Excluded Sites**")
                st.write(other_sites)



            blank() 


            if plot_route:  

                st.markdown("**Mapped Route**")

                simple_route = [fs_name] + res['route_sites'] + [fs_name]

                route_coords = []
                for loc in simple_route: 
                    if loc == fs_name: 
                        e = home_loc
                    else: 
                        e = tuple(sites.query(f"site_code == '{loc}'")[['lat', 'lng']].values[0]) 

                    route_coords.append(e)  

                m = folium.Map(location=home_loc , zoom_start=11)  

                s = 0 
                for i in range(len(route_coords) - 1):  
                    # take a to b coords 
                    if route_coords[i] != route_coords[i+1]: 
                        s += 1
                    folium.PolyLine([route_coords[i], route_coords[i+1]], 
                        popup=f"leg {s}", 
                        color='black',
                        weight=5,
                        dash_array='10', 
                        opacity=0.5).add_to(m) 

                for idx,stop in enumerate(route_coords):  
                    stop_name = simple_route[idx] 
                    if stop_name != fs_name:
                        html = f"""
                        <u><b>{stop_name}</b></u><br><br>
                        number of tasks: <b>{facility_dict[stop_name]['number of tasks']}</b><br>
                        task time est. : <b>{int(facility_dict[stop_name]['total time to complete'])}</b> min<br>
                        total points: <b>{facility_dict[stop_name]['total_points']}</b>
                        """   
                        wid = 210 
                        ht = 100
                    else: 
                        html = f"{stop_name}" 
                        wid = 100 
                        ht = 40 

                    iframe = folium.IFrame(html=html, width=wid, height=ht)
                    popup = folium.Popup(iframe, max_width=1000) 

                    color = 'green' if simple_route[idx] == fs_name else 'red'
                    folium.CircleMarker(
                        location=stop,
                        radius=5,
                        fill=True,
                        color=color,
                        fill_opacity=.9, 
                        popup=popup
                    ).add_to(m)

                st.markdown(m._repr_html_(), unsafe_allow_html=True)  
                
        else: 
            st.error("Invalid Password") 
            st.markdown("Contact Grant if you're having password issues")

with route_build_tab: 
    
    with st.form(key='f'): 
        
        fs_name = st.selectbox('Name', sorted(list(routes.keys())))   
        
        
        choice = st.radio(
            "Select change to make", 
            ["Schedule appointment (same day only)", 
             "Include Sites", 
             "Exclude site(s) from route",
             "Exclude task group(s) from route", 
             "Regenerate route with shortened day", 
             "Add Facility Block Task(s)"], 
            help="Note you can **only choose one** change to make at a time"
        )  
        # "Regenerate route with shortened day"
        
        #day = st.date_input('Select Date')  
        
        time_remaining = st.slider("Time Remaining in Day (hours)", min_value=1., max_value=8., value=8., step=.5, help='Leave at 8 if re-optimizing a full day') 
        
        c1, c2 = st.columns(2) 
        start_loc = c1.selectbox("Route Start Location", ['Home'] + sorted(list(sites['site_code'].values))) 
        end_loc = c2.selectbox("Route End", ['Home'] + sorted(list(sites['site_code'].values)), disabled=True)
        end_loc = "Home"         
        blank()
        
        with st.expander("Schedule Appointment"):
            #time_col, 
            site_col, appt_length_col = st.columns(2) 
            #.selectbox('Day of Week', ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri'])
            #t = time_col.time_input("Appointment Time", datetime.time(8, 0)) 
            rd = site_col.selectbox("Site", [''] + sorted(list(sites['site_code'].values)))  
            appt_length = appt_length_col.number_input("Appointment Length (minutes)", min_value=5, step=5, value=15)  
            appt_notes = st.text_input("Appointment Notes", help="any details you include here will display with your generated route")
#            start_time = st.time_input("Day Start", datetime.time(8, 0), help='if you want to select non-standard start time') 
#            custom_start_time = st.checkbox("Use custom start time", False, help="only check this if you're setting a non-standard start time") 
            hours_into_day = st.slider("Hours into day", min_value=1., max_value=8., value=0., step=.25, help='For example, this would be set to 4 if you have a 12 pm appt and start your day at 8. If its currently 2 pm and you have a 4 pm appt, this value would be set to 2.')


        with st.expander("Include Sites"):  
            rd_to_include = st.multiselect("Sites", [''] + sorted(list(sites['site_code'].values)))   
            st.caption("Selecting sites here will limit your route to tasks **only** at these properties")
            
        with st.expander("Exclude Sites"): 
            rd_to_exclude = st.multiselect("Site", [''] + sorted(list(sites['site_code'].values)))   
            st.caption("Selecting sites here will limit your route to tasks **not at these properties**")
        
        with st.expander("Exclude Task Groups"):  
            tg_to_exclude = st.multiselect("Task Group", [''] + list(sorted(task_groups['task group'].unique(), key=lambda x: x.split('--')[-1])) + [x['task group'] for x in fblocks])  
            st.caption("Any task groups selected here will be **excluded** from the route")
            
        with st.expander("Add Facility Block Tasks"): 
            new_fblock_sites = st.multiselect("Select site(s) to add block times at", [''] + sorted(list(sites['site_code'].values))) 
            st.caption("Note that added blocks **only affect today's route**, and are only added to the list of potential tasks. This means that, although while likely, you are **not** guaranteed to route to that facility just because you added a block time (use the add appointment feature in that case).")
        
        
        reopt_reason = st.selectbox("Re-optimization reason", 
                     ['', 'Break-In', 'Trash', 'Personal Reason', 
                    'Facility Incident (ex. water break, fire/police request, live in, combative tenant, etc.)', 
                    'Vehicle Incident (ex. flat tire)',  
                    'Gate Down', 
                    'Same Day Appointment'],  
                     help='Please list the reason for re-optimizing. This helps identify situations where we can improve the system and diagnose the main pain points.', 
                    format_func=lambda x: 'Select an option' if x == '' else x
                    )
        
        fs_password = st.text_input("Password") # type='password'  
        
        blank()
        st.info("**Note**: route generation can take up to 5 minutes – don't close page after pressing the button") 
        save_res = st.checkbox("Save Results", True)
        reopt = st.form_submit_button('Build New Route')  
        
    
    if reopt:    
        
        day = np.datetime64('today').item() 
        
        password_valid = password_authenticate(fs_password, PASSWORDS)  
        
        if password_valid: 
            st.success("Valid Password")  
            
            PROBLEM = False
            
#            # if an fs has pto this week, shorten their week 
#            if fs_name in pto_counts: 
#                N_days = n_weekdays - pto_counts[fs_name] 
#            else: # they have a normal week
#                N_days = n_weekdays 
                
#            st.write(f"fs_name in pto_counts: {fs_name in pto_counts} | {N_days}") 

            home_loc = fs_homes.get(fs_name.replace('  ', ' '), np.nan) 
            if pd.isnull(home_loc): 
                st.error(f"Missing a home location for {fs_name}")
            
            

            start_end_locs = {'start': None if start_loc == 'Home' else start_loc, 
                            'end': None if end_loc == 'Home' else start_loc, 
                            'start_is_rd': False if start_loc == 'Home' else True, 
                            'end_is_rd': False if end_loc == 'Home' else True} 
            
            if choice == "Schedule appointment (same day only)": 
                schedule_appt, exclude_site, exclude_task_group, shorten, add_fblocks, include_sites = True, False, False, False, False, False 
            elif choice == "Include Sites": 
                schedule_appt, exclude_site, exclude_task_group, shorten, add_fblocks, include_sites = False, False, False, False, False, True
            elif choice == "Exclude site(s) from route": 
                schedule_appt, exclude_site, exclude_task_group, shorten, add_fblocks, include_sites = False, True, False, False, False, False
            elif choice == "Regenerate route with shortened day": 
                schedule_appt, exclude_site, exclude_task_group, shorten, add_fblocks, include_sites = False, False, False, True, False, False
            elif choice == "Add Facility Block Task(s)": 
                schedule_appt, exclude_site, exclude_task_group, shorten, add_fblocks, include_sites = False, False, False, False, True, False
            else: 
                schedule_appt, exclude_site, exclude_task_group, shorten, add_fblocks, include_sites = False, False, True, False, False, False

            # CHECK PASSWORD   
            
            if time_remaining == 8:
                TMAX = 8.25 
            else: 
                TMAX = time_remaining    
                
            # TOP PRIORITY TASKS HEURISTIC    
            # this limits algo to only see tasks with highest priority 
            TOP_N_TASKS = 70  
            ONLY_TOP_TASKS = True 
                
            if include_sites: 
                pdat = prep_data.prepare_data_with_facility_blocks(fblocks, fs_name, fs_name, task_groups=task_groups, dtable=dtable, return_missing=True)
                task_list = [x for x in pdat['task_list'] if x['site_code'] in rd_to_include]  
                
                if ONLY_TOP_TASKS: 
                    task_list = get_top_tasks(task_list, TOP_N_TASKS)
                
                try:

                    with st.spinner(f"Generating route for tasks only at **{rd_to_include}**"): 
                        ga = algo.GA(
                            task_list=task_list, 
                            name=pdat['name'], 
                            dtable=dtable, 
                            tmax=TMAX*60*60, 
                            pop_size=param_dict['POP_SIZE'], 
                            block_time=param_dict['BLOCK_TIME'], 
                            start_end_locs=start_end_locs
                        ) 

                        ga.run(d2d=param_dict['D2D'], iters=param_dict['ITERS']) 

                        res = ga.output   
                        res.update({'reoptimized_route': True,  
                                    'reoptimized_timestamp': str(np.datetime64('now')), 
                                    'revision': {'type': 'include_sites', 'value': rd_to_include}}) 

                        if save_res:
                            st.success(f"Route successfully generated! Route generated for tasks at: {rd_to_include}") 
                            st.caption("Route will be updated in the 'Optimal Task Routes' tab soon")
                        else:
                            st.success("Route successfully generated! This is a test run, the route will not save permanantly")  
                            
                except Exception as e: 
                    print(e)
                    st.error("An error occured re-optimizing your route – try again with different inputs or contact Grant") 
                    PROBLEM = True
                
            if add_fblocks: 
                new_blocks = create_new_fblocks(new_fblock_sites) 
                fblocks += new_blocks  
                
                pdat = prep_data.prepare_data_with_facility_blocks(fblocks, fs_name, fs_name, task_groups=task_groups, dtable=dtable, return_missing=True)
                
                with st.spinner(f"Generating route for **{time_remaining}** hour day starting at **{start_loc}** and ending at **Home** and adding Facility Blocks at **{new_fblock_sites}**"): 
                                        
                    try:  
                        
                        if ONLY_TOP_TASKS: 
                            task_list = get_top_tasks(pdat['task_list'], TOP_N_TASKS) 
                        else: 
                            task_list = pdat['task_list']
                
                        ga = algo.GA(
                                task_list=task_list, 
                                name=pdat['name'], 
                                dtable=dtable, 
                                tmax=TMAX*60*60, 
                                pop_size=param_dict['POP_SIZE'], 
                                block_time=param_dict['BLOCK_TIME'], 
                                start_end_locs=start_end_locs
                            ) 

                        ga.run(d2d=param_dict['D2D'], iters=param_dict['ITERS']) 

                        res = ga.output   
                        res.update({'reoptimized_route': True,  
                                    'reoptimized_timestamp': str(np.datetime64('now')), 
                                    'revision': {'type': 'added fblocks', 'value': new_fblock_sites}}) 

                        if save_res:
                            st.success(f"Route successfully generated! Facility blocks added at **{new_fblock_sites}**") 
                            st.caption("Route will be updated in the 'Optimal Task Routes' tab soon")
                        else:
                            st.success("Route successfully generated! This is a test run, the route will not save permanantly")  
                            
                    except: 
                        st.error("An error occured re-optimizing your route – try again with different inputs or contact Grant") 
                        
                        PROBLEM = True 
                
            
            if shorten:   
                pdat = prep_data.prepare_data_with_facility_blocks(fblocks, fs_name, fs_name, task_groups=task_groups, dtable=dtable, return_missing=True)
                
                with st.spinner(f"Generating route for **{time_remaining}** hour day starting at **{start_loc}** and ending at **Home**"): 
                                        
                    try: 
                        
                        if ONLY_TOP_TASKS: 
                            task_list = get_top_tasks(pdat['task_list'], TOP_N_TASKS) 
                        else: 
                            task_list = pdat['task_list']
                
                        ga = algo.GA(
                                task_list=task_list, 
                                name=pdat['name'], 
                                dtable=dtable, 
                                tmax=TMAX*60*60, 
                                pop_size=param_dict['POP_SIZE'], 
                                block_time=param_dict['BLOCK_TIME'], 
                                start_end_locs=start_end_locs
                            ) 

                        ga.run(d2d=param_dict['D2D'], iters=param_dict['ITERS']) 

                        res = ga.output   
                        res.update({'reoptimized_route': True,  
                                    'reoptimized_timestamp': str(np.datetime64('now')), 
                                    'revision': {'type': 'shortened day', 'value': time_remaining}}) 

                        if save_res:
                            st.success(f"Route successfully generated! Day Shortened to **{time_remaining}** hours") 
                            st.caption("Route will be updated in the 'Optimal Task Routes' tab soon")
                        else:
                            st.success("Route successfully generated! This is a test run, the route will not save permanantly")  
                            
                    except: 
                        st.error("An error occured re-optimizing your route – try again with different inputs or contact Grant") 
                        
                        PROBLEM = True 
            
            if schedule_appt:
                
                #ptime = t.hour + (t.minute / 60)   
                
#                if custom_start_time: 
#                    #st.write(start_time, type(start_time), dir(start_time))  
#                    st.info("Using custom start time")
#                    cstart_time = start_time.hour + (start_time.minute / 60) 
#                
#                # day_int = day.weekday()    
#                day_int = day.weekday() 
#                try:
#                    start_time, stop_time = fs_workdays.loc[fs_name, f"{day_int}_am"], fs_workdays.loc[fs_name, f"{day_int}_pm"]  
#                except: 
#                    start_time, stop_time = fs_workdays.loc[fs_name.replace('  ', ' '), f"{day_int}_am"], fs_workdays.loc[fs_name.replace('  ', ' '), f"{day_int}_pm"]
#
#                # get time into day 
#                start_time, stop_time = np.datetime64(start_time).item(), np.datetime64(stop_time).item() 
#                start_time, stop_time = start_time.hour + (start_time.minute / 60), stop_time.hour + (stop_time.minute / 60)  
#                
#                if custom_start_time: 
#                    hours_into_day = ptime - cstart_time 
#                else:
#                    hours_into_day = ptime - start_time 
#                    
#                if hours_into_day < 0: 
#                    st.error("**Error**: appointment time must be within FS workday") 
                    
                #hours_into_day = appt_hrs_into_day
                    
                sched_params = {
                    'fs': fs_name, 
                    #'time': str(t), 
                    'hours_into_day': hours_into_day, 
                    'site': rd, 
#                    'custom_start_time': custom_start_time, 
#                    'start_time': start_time, 
#                    'stop_time': stop_time, 
                    'notes': appt_notes, 
                    'same_day_modification': True
                }

                

                # BUILD ROUTE  

                pdat = prep_data.prepare_data_with_facility_blocks(fblocks, fs_name, fs_name, task_groups=task_groups, dtable=dtable, return_missing=True) 
                
                if ONLY_TOP_TASKS: 
                    task_list = get_top_tasks(pdat['task_list'], TOP_N_TASKS) 
                else: 
                    task_list = pdat['task_list'] 
                
    
                try: 
                    with st.spinner(f"Generating route with appointment at **{rd} {hours_into_day}** hours into day with a **{TMAX}** hour day"):  
                        res = algo.schedule_appt( 
                            appt_site=rd, 
                            time_into_day=hours_into_day*60*60, 
                            total_hours=TMAX*60*60,
                            task_list=task_list, 
                            name=pdat['name'], 
                            dtable=dtable, appt_length=appt_length)  
                        if save_res:
                            st.success(f"Route successfully generated! Appointment added at **{rd}**") 
                            st.caption("Route will be updated in the 'Optimal Task Routes' tab soon")
                        else:
                            st.success("Route successfully generated! This is a test run, the route will not save permanantly")  

                        res.update({'revision': {'type': 'appointment', 'value': sched_params}})  
                        
                        
                except Exception as e:  
                    print(f"error with appt sched: {e}")
                    st.error("An error occured re-optimizing your route – try again with different inputs or contact Grant") 
                    PROBLEM = True

            if exclude_site: 
                
                pdat = prep_data.prepare_data_with_facility_blocks(fblocks, fs_name, fs_name, task_groups=task_groups, dtable=dtable, return_missing=True)
                task_list = [x for x in pdat['task_list'] if x['site_code'] not in rd_to_exclude]   
                
                if ONLY_TOP_TASKS: 
                    task_list = get_top_tasks(task_list, TOP_N_TASKS)
                
                try:

                    with st.spinner(f"Generating route and excluding **{rd_to_exclude}**"): 
                        ga = algo.GA(
                            task_list=task_list, 
                            name=pdat['name'], 
                            dtable=dtable, 
                            tmax=TMAX*60*60, 
                            pop_size=param_dict['POP_SIZE'], 
                            block_time=param_dict['BLOCK_TIME'], 
                            start_end_locs=start_end_locs
                        ) 

                        ga.run(d2d=param_dict['D2D'], iters=param_dict['ITERS']) 

                        res = ga.output   
                        res.update({'reoptimized_route': True,  
                                    'reoptimized_timestamp': str(np.datetime64('now')), 
                                    'revision': {'type': 'site_exclude', 'value': rd_to_exclude}}) 

                        if save_res:
                            st.success(f"Route successfully generated! Sites excluded: **{rd_to_exclude}**") 
                            st.caption("Route will be updated in the 'Optimal Task Routes' tab soon")
                        else:
                            st.success("Route successfully generated! This is a test run, the route will not save permanantly")  
                            
                except Exception as e: 
                    print(e)
                    st.error("An error occured re-optimizing your route – try again with different inputs or contact Grant") 
                    PROBLEM = True

            if exclude_task_group:  
                pdat = prep_data.prepare_data_with_facility_blocks(fblocks, fs_name, fs_name, task_groups=task_groups, dtable=dtable, return_missing=True)
                task_list = [x for x in pdat['task_list'] if x['task group'] not in tg_to_exclude]   
                
                if ONLY_TOP_TASKS: 
                    task_list = get_top_tasks(task_list, TOP_N_TASKS)
                
                try: 

                    with st.spinner(f"Generating route and excluding **{tg_to_exclude}**"): 
                        ga = algo.GA(
                            task_list=task_list, 
                            name=pdat['name'], 
                            dtable=dtable, 
                            tmax=TMAX*60*60, 
                            pop_size=param_dict['POP_SIZE'], 
                            block_time=param_dict['BLOCK_TIME'], 
                            start_end_locs=start_end_locs
                        ) 

                        ga.run(d2d=param_dict['D2D'], iters=param_dict['ITERS']) 

                        res = ga.output 
                        res.update({'reoptimized_route': True, 
                                    'reoptimized_timestamp': str(np.datetime64('now')), 
                                    'revision': {'type': 'task_group_exclude', 'value': tg_to_exclude}})  

                        if save_res:
                            st.success(f"Route successfully generated! Task groups excluded: **{tg_to_exclude}**") 
                            st.caption("Route will be updated in the 'Optimal Task Routes' tab soon")
                        else:
                            st.success("Route successfully generated! This is a test run, the route will not save permanantly")  
                            
                except: 
                    st.error("An error occured re-optimizing your route – try again with different inputs or contact Grant") 
                    PROBLEM = True

            
            if not PROBLEM:  
                
                reopt_reason = 'no reason selected' if reopt_reason == '' else reopt_reason
            
                res.update({'revision_time_remaining': time_remaining, 
                            'start_loc': start_loc, 
                            'end_loc': end_loc,  
                            'reopt_reason': reopt_reason
                           })

                facility_dict = create_facility_dict(res['route_full']) 
                site_path = create_site_level_path(res['route_sites'], facility_dict) 
                #site_path_markdown = format_site_level_path(site_path)    

                fitness = res['fitness'] 
                tor = manual_time_on_road_pct(fitness['travel_times']) 
                time_on_rd = tor['pct'] 
                drive_time = tor['drive_time']


                st.markdown('**Re-Optimized Route**')  
                ###---  
                sorder = preserve_order_unique_sites(res['route_sites']) 
                sorted_route = sort_route(res['route_full'], sorder)
                
                #sorted_route = sort_by_task_group(res['route_full'])
                st.write(f"Route will score {fs_name.split()[0]} **{fitness['score']}** task points in **{round(fitness['cost_less_transition_pen'], 2)}** hours, and include **{drive_time}** hours of driving (**{round(time_on_rd * 100)}%** of the day)") 
                blank()  

                last_site = None 
                i = 0

                md_list = []
                for s in sorted_route:  
                    md = "\n\n" 

                    tg = s['task group'].split('--')[0].strip() 
                    site = s['site_code'] 

                    if site != last_site:  
                        stop = site_path[i]  
                        md += f"{i+1}. <u>**{site}</u> *-- {int(stop['total time to complete'])} minutes of tasks -- {stop['total_points']} points collected***\n" 
                        last_site = site
                        i += 1

                    # check if unit num in task group 
                    if s['task group'].count('--') == 2:  
                        unit_num = s['task group'].split('--')[1].strip()  
                        redline_links = parse_redline_links(tg, site, tasks_raw, fs_name, unit_num=unit_num)
                    else: 
                        redline_links = parse_redline_links(tg, site, tasks_raw, fs_name)  

                    md += f"\t- {s['task group']}: {redline_links} -- *{int(round(s['experimental_priority']))} points* -- *{int(round(s['total time to complete']))} minutes*<br>"
                    md_list.append(md) 

                path_str = "".join(md_list)
                st.markdown(path_str, unsafe_allow_html=True)  
                blank()
                ###---
                
#                st.write(f"Route will score {fs_name.split()[0]} **{fitness['score']}** task points in **{round(fitness['cost_less_transition_pen'], 2)}** hours, and include **{drive_time}** hours of driving (**{round(time_on_rd * 100)}%** of the day)") 
                
                if start_loc != 'Home': 
                    st.write(f"**Note** – this route starts at {start_loc} and **not** at home location")
                
#                for s in site_path_markdown: 
#                    st.markdown(s) 
#                    blank()

                note_tables = get_task_notes(task_group_route=res['route_task_groups'], 
                               full_tasks=tasks_raw, 
                               name=fs_name) 

#
#                with st.expander("Route Table"): 
#                    table = create_route_table(res['route_full']) 
#                    st.write(table) 
#
#                with st.expander("Redline Links"): 
#                    for s in res['route_full']:  
#                        tg = s['task group'].split('--')[0].strip() 
#                        site = s['site_code']  
#                        redline_links = parse_redline_links(tg, site, tasks_raw, fs_name)  
#                        st.markdown(f"**{s['task group']}**: {redline_links}", unsafe_allow_html=True)
#                        #st.markdown(redline_links) 
#                        blank()

                if len(note_tables.keys()) > 0: 
                    with st.expander("Supplementary Task Notes"):
                        for task in note_tables:  
                            st.markdown(f"**{task}**")
                            st.write(note_tables[task]) 

                ttimes = travel_times_markdown(res['route_sites'], fitness['travel_times'])  
                with st.expander("Travel Times"): 
                    st.info("**Note**: a **20 minute** buffer time was added to all travel times here, **with the exception of the final leg of the day**, to account for parking upon new facility arrival, traffic delays, etc.")
                    st.markdown(ttimes) 

                with st.expander("Excluded Task Groups"): 
                    visited_sites = set(res['route_sites'])
                    unvisited_df = (
                        pd.DataFrame(res['unvisited'])
                        .drop(['visited', 'task_ids'], axis=1)
                        .set_index('task group') 
                        .rename(columns={'experimental_priority': 'points', 'total time to complete': 'time to complete'}) 
                    )  

                    unvisited_df['time to complete'] = unvisited_df['time to complete'].apply(lambda x: str(int(x)) + ' min')
                    unvisited_df['points'] = unvisited_df['points'].astype(int)

                    for site in visited_sites:
                        st.markdown(f"**{site}**") 
                        site_df = unvisited_df.query(f"site_code == '{site}'")
                        if site_df.empty: 
                            st.write(f"All tasks picked up at {site}") 
                        else:
                            st.write(site_df) 

                    other_sites = unvisited_df[~unvisited_df['site_code'].isin(visited_sites)]

                    st.markdown("**Excluded Sites**")
                    st.write(other_sites)

                # generate summary sheet 

                # SAVE ROUTE TO AWS  
                # take routes fpath from aws, update current dict with new route, and re-save to aws 
                # also save appt adjustments to appt folder in s3  

                if save_res: 
        #            routes.update({fs_name: res})  

                    data_grabber.upload_file_to_s3(
                        data={fs_name: res}, 
                        bucket='fs-optimization', 
                        path='routes/modifications/', 
                        fname=f"{str(day)}_{fs_name}", 
                        MASTER_ACCESS_KEY=MASTER_ACCESS_KEY, 
                        MASTER_SECRET=MASTER_SECRET, 
                        file_type='pkl'
                    ) 

                    st.success("Route saved to database")

                plot_route = start_loc == 'Home' 
                
                if plot_route:  

                    st.markdown("**Mapped Route**")

                    simple_route = [fs_name] + res['route_sites'] + [fs_name]

                    route_coords = []
                    for loc in simple_route: 
                        if loc == fs_name: 
                            e = home_loc
                        else: 
                            e = tuple(sites.query(f"site_code == '{loc}'")[['lat', 'lng']].values[0]) 

                        route_coords.append(e)  

                    m = folium.Map(location=home_loc , zoom_start=11)  

                    s = 0 
                    for i in range(len(route_coords) - 1):  
                        # take a to b coords 
                        if route_coords[i] != route_coords[i+1]: 
                            s += 1
                        folium.PolyLine([route_coords[i], route_coords[i+1]], 
                            popup=f"leg {s}", 
                            color='black',
                            weight=5,
                            dash_array='10', 
                            opacity=0.5).add_to(m) 

                    for idx,stop in enumerate(route_coords):  
                        stop_name = simple_route[idx] 
                        if stop_name != fs_name:
                            html = f"""
                            <u><b>{stop_name}</b></u><br><br>
                            number of tasks: <b>{facility_dict[stop_name]['number of tasks']}</b><br>
                            task time est. : <b>{int(facility_dict[stop_name]['total time to complete'])}</b> min<br>
                            total points: <b>{facility_dict[stop_name]['total_points']}</b>
                            """   
                            wid = 210 
                            ht = 100
                        else: 
                            html = f"{stop_name}" 
                            wid = 100 
                            ht = 40 

                        iframe = folium.IFrame(html=html, width=wid, height=ht)
                        popup = folium.Popup(iframe, max_width=1000) 

                        color = 'green' if simple_route[idx] == fs_name else 'red'
                        folium.CircleMarker(
                            location=stop,
                            radius=5,
                            fill=True,
                            color=color,
                            fill_opacity=.9, 
                            popup=popup
                        ).add_to(m)

                    st.markdown(m._repr_html_(), unsafe_allow_html=True)   
                                    
            
        else: 
            st.error("Invalid Password")
            st.markdown("Contact Grant if you're having password issues")  
            
with pto_tab: 
    
    st.info("This tab is to be used **any time** where you won't or did not follow the system-generated route")
    
    with st.form(key='pto-form'): 
        fs_name = st.selectbox('Name', sorted(list(routes.keys()))) 
        day = st.date_input('Select Date')  
        fs_password = st.text_input("Password")
        submit_pto = st.form_submit_button("Submit") 
        
    if submit_pto:  
        
        password_valid = password_authenticate(fs_password, PASSWORDS)
        
        if password_valid:
        
            day = str(day)
            pto_file = {
                'name': fs_name, 
                'date': day, 
                'submitted_by': 'fs'
            }  

            data_grabber.upload_file_to_s3(data=pto_file, 
                      bucket='fs-optimization', 
                      path=f"routes/pto/{day}/", 
                      fname=fs_name, 
                      MASTER_ACCESS_KEY=MASTER_ACCESS_KEY, 
                      MASTER_SECRET=MASTER_SECRET, 
                      file_type='json') 

            st.success(f"Success, submitted PTO for **{day}**") 
        else: 
            st.error("Invalid Password") 
            
            
with appt_tab: 
        
    st.info("Next day appointments need to be submitted by **5 pm MT** / **6 pm CT** / **7 pm ET** the day before")
    
    with st.form(key='appt-form'):  
        
        appt_c1, appt_c2 = st.columns(2)
        fs_name = appt_c1.selectbox('Name', [''] + sorted(list(routes.keys())))   
        day = appt_c2.date_input('Appointment Date') 
        
        time_col, site_col, appt_length_col = st.columns(3) 
        #.selectbox('Day of Week', ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri'])
        t = time_col.time_input("Appointment Time", datetime.time(8, 0)) 
        rd = site_col.selectbox("Site", [''] + sorted(list(sites['site_code'].values)) + ['Not Site Specific'])  
        appt_length = appt_length_col.number_input("Appointment Length (minutes)", min_value=5, step=5, value=15)  
        appt_notes = st.text_input("Appointment Notes", help="any details you include here will display with your generated route")
        start_time = st.time_input("Day Start Time", datetime.time(8, 0), help="select when you'll be starting your day") 
        #custom_start_time = st.checkbox("Use custom start time", False, help="only check this if you're setting a non-standard start time") 
        
        tz = st.selectbox("Timezone", [
            'Same Timezone', 
            'RD + 1 hour ahead', 
            'RD - 1 hour behind'
        ], help='change the selection here if the RD is in a different time zone than the home location')

        st.markdown('----') 
        fs_password = st.text_input("Password")
        submit_appt = st.form_submit_button("Submit")  

    if submit_appt: 
        password_valid = password_authenticate(fs_password, PASSWORDS) 
        today = str(np.datetime64('today'))
        
        if password_valid: 
            st.success("Valid Password")  
            
            ptime = t.hour + (t.minute / 60)   
            parsed_start_time = start_time.hour + (start_time.minute / 60) 
            hours_into_day = ptime - parsed_start_time  
            
            if tz == 'RD + 1 hour ahead': 
                hours_into_day -= 1
            
            if tz == 'RD - 1 hour behind': 
                hours_into_day += 1

            sched_params = {
                    'fs': fs_name, 
                    'time': str(t), 
                    'hours_into_day': hours_into_day, 
                    'site': rd, 
                    'start_time': str(start_time), 
                    'appt_length': appt_length, 
                    'notes': appt_notes, 
                    'submission_date': today,  
                    'scheduled_by': fs_name, 
                    'date': str(day)
                }
        
            now = np.datetime64('now') 
            
            data_grabber.upload_file_to_s3(data=sched_params, 
                      bucket='fs-optimization', 
                      path=f"routes/appointments/{day}/", 
                      fname=f"{fs_name}_{str(now)}", 
                      MASTER_ACCESS_KEY=MASTER_ACCESS_KEY, 
                      MASTER_SECRET=MASTER_SECRET, 
                      file_type='json') 

            st.success(f"Success, submitted appointment for **{fs_name}** on **{day}**") 
        else: 
            st.error("Invalid Password") 