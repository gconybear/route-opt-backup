import pandas as pd 
import numpy as np 
from itertools import tee
import copy  
import networkx as nx 
import random 
from python_tsp.exact import solve_tsp_dynamic_programming 
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing 
import time 
import fast_tsp

class GA: 
    
    def __init__(self, task_list, name, dtable, tmax, 
                 pop_size, block_time, time_buffer=20*60, 
                 facility_check_ins=None, max_sites=None, 
                 start_end_locs={'start': None, 'end': None, 'start_is_rd': False, 'end_is_rd': False}):  
        
        # new - variable start and end locations 
        if start_end_locs['start'] is None:  
            # we just use home location as usual 
            self.start_loc = name   
            self.start_is_rd = False   
            self.custom_locs = False
        else: 
            self.start_loc = start_end_locs['start'] 
            self.start_is_rd = start_end_locs['start_is_rd'] 
            self.custom_locs = True
        
        if start_end_locs['end'] is None: 
            self.end_loc = name  
            self.end_is_rd = False 
            self.custom_locs = False
        else: 
            self.end_loc = start_end_locs['end']  
            self.end_is_rd = start_end_locs['end_is_rd']
            self.custom_locs = True
            
        # get filtered dtable  
        
        def filter_df(df, site_list):
            mask = df['start'].isin(site_list) | df['end'].isin(site_list)
            return df[mask] 
 
        
        unique_stops = set([x['site_code'] for x in task_list] + [name]) 
        dtable = filter_df(dtable, unique_stops)
        
#        m1 = dtable['start'].isin(unique_stops)
#        m2 = dtable['end'].isin(unique_stops)
        
        self.og_task_list = task_list
        self.task_list = copy.deepcopy(task_list)
        self.name = name 
        self.dtable = dtable # dtable[m1 | m2] 
        self.tmax = tmax 
        self.pop_size = pop_size 
        self.BLOCK_TIME = block_time  
        self.TIME_BUFFER = time_buffer  
        self.MAX_SITES = max_sites if max_sites is not None else np.inf 
        self.runtimes = [] 
        
        # precompute distance matrix on initialization  
        # every dmat creation hereafter will just filter this down  
        print("precomputing full distance matrix...") 
        site_appends = [{'site_code': site, 'task group': site} for site in set([x['site_code'] for x in self.task_list])]
        self.full_dmat, self.full_dmat_cols = self.create_distance_matrix_og(self.task_list + site_appends)
        
        if facility_check_ins is not None:  
            cblocks = self.make_check_in_blocks(sites=facility_check_ins) 
            self.task_list += cblocks 
            self.check_in_blocks = cblocks 
        else: 
            self.check_in_blocks = None  
            
        print(f"algo initialized: custom start and end locs --> {self.custom_locs}") 
        
    def get_distance(self, start: str, end: str): 
        
        try: 
            start_idx, end_idx = self.full_dmat_cols.index(start), self.full_dmat_cols.index(end)   
             
        except Exception as e:  
            print(f"failed to get distance: {e}")
            print(f"one of {start}, {end} not found in distance matrix") 
            return np.nan
        
        return self.full_dmat[start_idx, end_idx]
    
    def make_check_in_blocks(self, sites:list[tuple[str, float]], t=15.):  
        
        """
        sites: list of (site_code, priority) tuples 
        t: time in minutes
        """

        blocks = []
        for site, p in sites: 
            blocks.append({'task group': f'Facility Check In -- {site}',
          'number of tasks': 1,
          'total time to complete': t,
          'experimental_priority': p,
          'site_code': site,
          'visited': False}) 

        return blocks 
    
    def get_route(self, task_list): 
        return [x for x in task_list if x['visited']]
    
    def pairwise(self, iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b) 
    
    def fitness(self, route, initial_pop=False, output=False):   
        
        """
        route: list of dicts (MAKE SURE ONLY NODES WITH 'VISITED' key == TRUE) 
        
        incorporates facility block time 
        """
        
        if not initial_pop: 
            assert all(x['visited'] for x in route)

        SCORE_NAME = 'experimental_priority' 
        COST_NAME = 'total time to complete' 
        CONVERSION_FACTOR  = 60 # need to make everything seconds // if cost is minutes * 60 

        # --- sum all task priorities ---
        task_scores = sum([x[SCORE_NAME] for x in route])

        # --- sum all task costs ---
        task_costs = sum([x[COST_NAME] * CONVERSION_FACTOR for x in route])

        # --- get all travel pairs in route --- 

        # 1. add home to beg and end  
        modified_route = [{'site_code': self.start_loc, 'task group': self.start_loc}] + route + [{'site_code': self.end_loc, 'task group': self.end_loc}] 

        # 2. sum all travel pairs and += costs  
        travel_costs = 0  
        transitions = 0  
        travel_times = []
        for a,b in self.pairwise(modified_route):   
            
            # check to see if new facility is being visited
            if (a['site_code'] != b['site_code']) and (b['site_code'] != self.end_loc): 
                transitions += 1 
                diff_facility = 1 
            else: 
                diff_facility = 0 
            
            # find distance between a and b  
            dist = self.get_distance(a['task group'], b['task group']) 
            
            if np.isnan(dist): 
                print(f"no dist between {a['task group'], b['task group']}") 
            
            travel_time = dist + (diff_facility * self.TIME_BUFFER)
            travel_costs += travel_time  
            
            if output: 
                travel_times.append(travel_time / 60) # report it in minutes 
                 
#            row = self.dtable.query(f"start == '{a['site_code']}' and end == '{b['site_code']}'")  
#            if not row.empty:   
#                
#                travel_time = row['value'].values[0] + (diff_facility * self.TIME_BUFFER)
#                travel_costs += travel_time  
#                
#                if output: 
#                    travel_times.append(travel_time / 60) # report it in minutes 
#            else: 
#                raise f"distance between {a['site_code']} and {b['site_code']} not found" 
                
                
        facility_transition_penalty = transitions * self.BLOCK_TIME
         
        total_cost = task_costs + travel_costs + facility_transition_penalty 
        
        if transitions > self.MAX_SITES:
            total_cost *= total_cost
        
        return {'score': task_scores, 'cost': total_cost, 
                'travel_cost': travel_costs, 'time_costs': task_costs, 
                'transition_pen': facility_transition_penalty, 'travel_times': travel_times} 

    
    def point_ommission_probability(self, maxloop=100):  

        N = len(self.task_list)
        task_copy = copy.deepcopy(self.task_list)
        c = 0 
        loop = 0 

        while loop < maxloop: 

            rN = np.random.choice(np.arange(N), 1)[0]

            rL_low, rL_high = tuple(sorted(np.random.choice(np.arange(N), 2, replace=False))) 
            rL = np.arange(rL_low, rL_high+1)

            rS = rL[:rN] 

            # route cost, not length of list!           
            if len(rS) == 0: 
                c += 1 
            else:
                rS = [task_copy[i] for i in rS]  
                for t in rS: 
                    t['visited'] = True
                
                drs = self.fitness(rS)['cost']
                if drs <= self.tmax: 
                    c += 1

            loop += 1

        return 1 - (c / maxloop)  
    
    def point_inclusion_prob(self): 
        
        tsp_solution = self.tour_improvement(self.task_list)  
        tsp_cost = tsp_solution['cost']['total']
        #tsp_cost = self.fitness(tsp_solution['route'])['cost']
        
        return np.sqrt(self.tmax / tsp_cost) 

        #return 55 / len(self.task_list)
    
    def create_initial_pop(self):   
        
        pop_size=self.pop_size
        
#         omission_prob = self.point_ommission_probability() 
#         self.omission_prob = omission_prob 
        
        inclusion_prob = self.point_inclusion_prob() 
        self.inclusion_prob = inclusion_prob

        p = 0 
        pop = []
        while p < pop_size:  

            tasks = copy.deepcopy(self.task_list) 
            np.random.shuffle(tasks)

            # create mask 
            mask = np.random.choice(np.arange(0,1, .001), size=len(tasks)) < inclusion_prob # this was > omission prob   
            if np.all(mask == False): 
                # choose two indices at random to set to true 
                m1, m2 = tuple(np.random.choice(np.arange(mask.shape[0]), size=2, replace=False)) 
                mask[m1] = True 
                mask[m2] = True 

            for i in range(mask.shape[0]): 
                tasks[i]['visited'] = mask[i] 

            pop.append(tasks)

            p += 1 

        return pop 
    
    # new crossover  
    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def crossover(self, r1: list, r2: list):    
        
#         assert all(x['visited'] for x in r1) 
#         assert all(x['visited'] for x in r2) 

        # check for exceptions -- return clone of random route 
        if (len(r1) == 1) or (len(r2) == 1) or (r1 == r2):  
            print('exception... crossover being skipped and one route being cloned at random')
            return {'route_table': np.nan, 'route': random.choice([r1, r2])}
        
        home_name = self.name
        r1 = [self.start_loc] + r1 + [self.end_loc]
        r2 = [self.start_loc] + r2 + [self.end_loc]

        # 1. create nx graphs from both routes 
        r1_tuples = []
        for p in self.pairwise(r1): 
            r1_tuples.append(p) 

        r2_tuples = []
        for p in self.pairwise(r2): 
            r2_tuples.append(p)     

        G1 = nx.Graph()
        G1.add_edges_from(r1_tuples) 
        G2 = nx.Graph()
        G2.add_edges_from(r2_tuples) 

        # 2. create parental graph - EDIT: don't think we need this 
        # P = nx.compose(G1, G2) 

        # 3. find common nodes 
        common = set(list(G1.nodes)).intersection(list(G2.nodes))   
        if len(common) == 1: 
            return {'route_table': np.nan, 'route': random.choice([r1, r2])}

        # 4. find intermediate paths 
        ipaths = {n:{} for n in common}
        degrees = {} 
        for n1 in common:  
            exp_degrees_sub = {}
            for n2 in common:  
                # looking at all pairs of common nodes 
                if n1 != n2:    
                    other_common = [x for x in common if x not in [n1, n2]] 
                    paths = [] 

                    # check both graphs for routes and append route if it doesn't go through any other common nodes
                    for p in nx.all_simple_paths(G1, source=n1, target=n2): 
                        if not any(x in other_common for x in p):  
                            paths.append(p) 

                    for p in nx.all_simple_paths(G2, source=n1, target=n2): 
                        if not any(x in other_common for x in p):  
                            paths.append(p)

                    if len(paths) > 0:
                        ipaths[n1].update({n2: paths})  

                        exp_degrees_sub[n2] = len(paths)

            #degrees[n1] = len(ipaths[n1].keys()) 
            degrees[n1] = exp_degrees_sub


        # 5. run crossover algorithm 
        route = [] 
        remaining_common = copy.deepcopy(list(common))
        i = 0 
        try: 
            next_node = list(common)[0]  
        except IndexError: 
            return {'route_table': np.nan, 'route': random.choice([r1, r2])} 
        
        
        while 1:    
            if len(remaining_common) == 0:  
                # join last node to home node 
                elem = next_node, list(common)[0], [next_node, list(common)[0]] 
                route.append(elem)
                break  



            n = next_node 
            connected = ipaths[n].keys() 


            # next node is one with smallest degree  
            node_degrees_sorted = sorted(degrees[n].items(), key=lambda x: x[1])   
            node_degrees_sorted = [x for x in node_degrees_sorted if x[0] in remaining_common]
            candidate_nodes = [x for x in node_degrees_sorted if x[1] == node_degrees_sorted[0][1]]  

            if len(candidate_nodes) > 0:

                # if this is len(1) then it chooses the sole element else a random choice  
                start_node = next_node
                next_node = random.choice(candidate_nodes)[0] 
                next_path = random.choice(ipaths[n][next_node]) 

                elem = start_node, next_node, next_path  
                route.append(elem)  

                # remove from eligible 
                remaining_common.remove(next_node)  

                if i == 0: 
                    #print(common, '--', remaining_common, home_name)  
                    # only remove it if start loc is NOT a site
                    if (self.start_loc in remaining_common) and (not self.start_is_rd):
                        remaining_common.remove(self.start_loc) # whatever the home equivalent is 
                    # only remove it if end loc is NOT a site
                    if (self.end_loc in remaining_common) and (not self.end_is_rd):
                        remaining_common.remove(self.end_loc)
            else:   
                # if there are non-visited common nodes  
                if len(remaining_common) > 0: 
                    # Select randomly a non-visited common node and insert it on the route after the current node   
                    start_node = next_node
                    next_node = random.choice(remaining_common)  
                    # just gets the simple path 
                    try:
                        next_path = [route[i-1][1], next_node]   
                    except IndexError: 
                        next_path = [route[i][1], next_node]
                    elem = start_node, next_node, next_path  
                    route.append(elem)   

                    remaining_common.remove(next_node)
                else: 
                    print('all crossover checks failed ', [x for x in connected if x not in route]) 


            i += 1     

        route_formatted = list(dict.fromkeys(self.flatten([x[-1] for x in route]))) 

        return {'route_table': route, 'route': route_formatted} 

    def order_task_list(self, order: list): 

        """
        this function takes in a list of task group names, and a route or task list (list of dicts) 
        and returns the list of dicts in the order 
        """  
        
        order = [x for x in order if x not in [self.start_loc, self.end_loc]]
        
#         if (not self.start_is_rd) and (not self.end_is_rd):   
#             # proceed as usual
#             order = [x for x in order if x != self.name] 
#         else:   
#             # if start is rd but not home, just exclude end loc 
#             if (self.start_is_rd) and (not self.end_is_rd): 
#                 order = [x for x in order if x != self.end_loc]  
            
#             # if end is rd but not start, just exclude start loc 
#             elif (not self.start_is_rd) and (self.end_is_rd): 
#                 order = [x for x in order if x != self.start_loc] 
#             else:
#             # if both, then pass 
#                 pass 
            
                
        ordered_route_list = [] 
        for stop in order: 
            ordered_route_list.append([x for x in self.task_list if x['task group'] == stop][0]) 

        for stop in ordered_route_list: 
            stop['visited'] = True 

        return ordered_route_list 

    def create_route_from_task_list(self, r): 
        full_route_list = copy.deepcopy(self.task_list)
        route = copy.deepcopy(r)
        for stop in route: 
            stop['visited'] = True  
            
        current_task_groups = [x['task group'] for x in route]
        for stop in full_route_list: 
            if stop['task group'] not in current_task_groups: 
                route.append(stop) 
                current_task_groups += [stop['task group']]

        return route  
    
    def count_adjacencies(self, alist): 
        counter = 0 
        pairs = []
        for i in range(len(alist)-1):
            if abs(alist[i] - alist[i+1]) == 1: 
                e = alist[i], alist[i+1]
                pairs.append(e)
                counter+=2  

        return counter, pairs 
    
    """
    add cost(pair, new node) --> distance(pair[0] + new node + pair[1]) - distance(pair[0], pair[1]) 
    add value(new node) --> new node score / add_cost
    """  
    
    def add_cost(self, pair: tuple, new_node: dict, home_node=False):   

        """
        essentially measuring the marginal cost of the additional task added 

        marginal cost = travel cost + time cost 
        """

        CONVERSION_FACTOR = 60 

        if home_node != False: 

            # its a dict 

            r1 = [home_node['pair'][0], new_node['site_code'], home_node['pair'][1]] 
            r2 = [home_node['pair'][0], home_node['pair'][1]]  

        else:
            r1 = [pair[0]['site_code'], new_node['site_code'], pair[1]['site_code']] 
            r2 = [pair[0]['site_code'], pair[1]['site_code']]
        
        # amount of unique sites visited * block time 
        """
        with variable start and end locations 
        
        if start is home, end is rd --> remove just start and count 
        if end is home, start is rd --> remove just end and count 
        if start and end are home --> count like normal  
        if start and end are rd --> don't filter list at all and count 
        """ 
        
        if (not self.end_is_rd) and (not self.start_is_rd): 
            # count like normal 
            r1_block_cost = (len(set([x for x in r1 if x != self.name])) - 1) * self.BLOCK_TIME
            r2_block_cost = (len(set([x for x in r2 if x != self.name])) - 1) * self.BLOCK_TIME  
        elif (not self.start_is_rd) and (self.end_is_rd): 
            # remove just start loc (start loc will be name so this works)
            r1_block_cost = (len(set([x for x in r1 if x != self.name])) - 1) * self.BLOCK_TIME
            r2_block_cost = (len(set([x for x in r2 if x != self.name])) - 1) * self.BLOCK_TIME  
        elif (self.start_is_rd) and (not self.end_is_rd):  
            # remove just end loc (end loc will be name so this works)
            r1_block_cost = (len(set([x for x in r1 if x != self.name])) - 1) * self.BLOCK_TIME
            r2_block_cost = (len(set([x for x in r2 if x != self.name])) - 1) * self.BLOCK_TIME  
        else:  
            # don't filter and just count 
            r1_block_cost = (len(set(r1)) - 1) * self.BLOCK_TIME
            r2_block_cost = (len(set(r2)) - 1) * self.BLOCK_TIME

        r1_cost = r1_block_cost
        for a,b in self.pairwise(r1):  
            
            if (a != b) and (b != self.end_loc): # switching fac and second fac is not home loc
                diff_facility = 1 
            else: 
                diff_facility = 0
            
            # find distance between a and b , lookup on dtable 
            row = self.dtable.query(f"start == '{a}' and end == '{b}'") 
            if not row.empty: 
                r1_cost += row['value'].values[0] + (diff_facility * self.TIME_BUFFER)
            else: 
                raise f"distance between {a} and {b} not found"  

        r1_cost += new_node['total time to complete'] * CONVERSION_FACTOR # IMPORTANT 

        r2_cost = r2_block_cost
        for a,b in self.pairwise(r2):  
            
            if (a != b) and (b != self.end_loc): # switching fac and second fac is not home loc
                diff_facility = 1 
            else: 
                diff_facility = 0
            
            # find distance between a and b , lookup on dtable 
            row = self.dtable.query(f"start == '{a}' and end == '{b}'") 
            if not row.empty: 
                r2_cost += row['value'].values[0] + (diff_facility * self.TIME_BUFFER)
            else: 
                raise f"distance between {a} and {b} not found" 

        return max(r1_cost - r2_cost, 0.00000001) # we don't want inf's in add value


    def add_value(self, pair: tuple, new_node: dict, home_node=False): 
        
        cost = self.add_cost(pair, new_node, home_node=home_node) 

        return {'value': new_node['experimental_priority'] / cost, 'cost': cost} 
    
    def sorter(self, column, site_order:list):
        """Sort function"""
        correspondence = {site: order for order, site in enumerate(site_order)}
        return column.map(correspondence)

    def add_operator(self, node_dict: dict, route: list[dict]): 

        """
        add operator acts on a specific node in a route 
        1. find the 3 nearest visited nodes (make sure to have an exception)
        2. compute add cost via logic  
        """ 
        assert all(x['visited'] for x in route)   
        
        node = node_dict['task group'] 
        node_site = node_dict['site_code']

        # get site codes from route 
        sites = [x['site_code'] for x in route] 
        candidates = [x['site_code'] for x in route if x['task group'] != node] 

        # find closest 3 
        closest = (
            self.dtable
            [(self.dtable['start'] == node_site) & (self.dtable['end'].isin(candidates))]
            .sort_values('value')
            .iloc[:3] 
            ['end'].values.tolist()
        )  

        adj = [(i,x) for (i,x) in enumerate(route) if x['site_code'] in closest and x['task group'] != node] 
        
        if len(closest) == 3:  
            if len(adj) > 3: 

                # trim list down 

                rdf = pd.DataFrame(route) 
                rdf = rdf[(rdf['site_code'].isin(closest)) & (rdf['task group'] != node)] 
                rdf = rdf.sort_values(by='site_code', key=lambda x: self.sorter(x, site_order=closest))

                adj = []
                for idx, row in rdf.head(3).iterrows(): 
                    e = (idx, dict(row)) 
                    adj.append(e)

        # calculate num adjacencies  
        acount, pairs = self.count_adjacencies([x[0] for x in adj])      

        # compute add cost and value based on adjacent pairs
        if acount == 0:  
            # test all 6 combinations 
            all_feasbilities = []
            for idx, nd in adj:  
                # create pair 
                if idx == 0: # idx at beg of route 

                    # pair should be (home, idx[0]) 
                    home_node = {'start': True, 'pair': (self.start_loc, 
                                                         nd['site_code'])} 

                    feas = self.add_value(pair=None, new_node=node_dict, home_node=home_node)  

                    all_feasbilities.append({'insertion_point': home_node['pair'], 'feas': feas})

                elif idx + 1 == len(route): # idx at end of route 

                    # pair should be (idx[-1], home) 
                    home_node = {'start': False, 'pair': (nd['site_code'], 
                                                           self.end_loc)} 

                    feas = self.add_value(pair=None, new_node=node_dict, home_node=home_node)  

                    all_feasbilities.append({'insertion_point': home_node['pair'], 'feas': feas})

                else:   

                    # do before AND after each idx 

                    # -- before -- 
                    s1 = [x for (i,x) in enumerate(route) if i == (idx - 1)][0]
                    s2 = nd 

                    dict_pair = tuple((s1, s2)) 
                    feas = self.add_value(pair=dict_pair, new_node=node_dict)    

                    before = {'insertion_point': dict_pair, 'feas': feas} 
                    all_feasbilities.append(before)

                    # -- after -- 
                    s3 = [x for (i,x) in enumerate(route) if i == (idx + 1)][0] 

                    dict_pair = tuple((s2, s3)) 
                    feas = self.add_value(pair=dict_pair, new_node=node_dict)   

                    after = {'insertion_point': dict_pair, 'feas': feas} 
                    all_feasbilities.append(after) 



        elif acount == 2: 
            # test insertion between the single pair 
            # add_value(pair=pairs[0], new_node=) 

            s1 = [x for (i,x) in enumerate(route) if i == pairs[0][0]][0]
            s2 = [x for (i,x) in enumerate(route) if i == pairs[0][1]][0]  

            dict_pair = tuple((s1, s2))  
            feas = self.add_value(pair=dict_pair, new_node=node_dict)

            all_feasbilities = [{'insertion_point': dict_pair, 'feas': feas}]

        else: 
            # test insertion between every pair and take one with minimal add cost   
            all_feasbilities = []
            for pair in pairs: 
                s1 = [x for (i,x) in adj if i == pair[0]][0] 
                s2 = [x for (i,x) in adj if i == pair[1]][0] 

                feas = self.add_value(pair=(s1, s2), new_node=node_dict) 
                all_feasbilities.append({'insertion_point': (s1, s2), 'feas': feas})  


        return all_feasbilities 
    
    def mutation(self, og_route, verbose=True): 
    
        route = copy.deepcopy(og_route)

        assert all(x['visited'] for x in route)

        # select node from task list at random
        node = random.choice(self.task_list)
        node_task = node['task group']  

        # if node in route, remove and connect adjacent nodes 
        in_route = len([x for x in route if x['task group'] == node_task]) == 1  
        
        if verbose:
            print(f"mutating with {'node removal' if in_route else 'node addition'}")

        if in_route: 
            revised_route = [x for x in route if x['task group'] != node_task]  

            return revised_route
        else: 
            # insert node using add operator 
            add_output = self.add_operator(node, route) 

            high_score = sorted(add_output, key=lambda x: x['feas']['value'], reverse=True)[0]['feas']['value'] 
            cands = [x for x in add_output if x['feas']['value'] == high_score] 

            # randomly select 
            insertion_point = random.choice(cands)['insertion_point']  
            node['visited'] = True 
            
            if insertion_point[-1] == self.end_loc: 
                route = route + [node] 
            elif insertion_point[0] == self.start_loc: 
                route = [node] + route
            
#             if self.name in insertion_point: 
#                 if insertion_point[-1] == self.name: # new node goes at very end 
#                     route = route + [node] 
#                 else: # new node goes at the beg 
#                     route = [node] + route 

            else:  
                idx2 = route.index(insertion_point[1]) 
                route.insert(idx2, node) 

            return route  
        
    def reg_gen(self, pop, fits, verbose=True): 
        """
        implement the mod != 0 case here 
        
        pop is list of candidates 
        fits is list of fitness dicts for each candidate 
        """  
        
        # print(f"avg gen fitness: {np.mean([x['score'] for x in fits])}")
        
        sorted_fits = sorted(list(zip(list(np.arange(len(pop))), pop, fits)), key=lambda x: x[-1]['score'])
        worst = sorted_fits[0]
        worst_score = worst[-1]['score']  
        worst_idx = worst[0]   
        
        
        if verbose:
            best = sorted_fits[-1][-1] 
            best_route_len = len(sorted_fits[-1][-2])
            best_score = round(best['score'], 2) 
            best_cost = round(best['cost'] / 60**2, 2)  

            avg_score = round(np.mean([x['score'] for x in fits]), 2)  
            avg_cost = round(np.mean([x['cost'] for x in fits]) / 60**2, 2) 
            
            print(f"best route length: {best_route_len} | best route score: {best_score} | best route cost: {best_cost} | avg score: {avg_score} | avg cost: {avg_cost} | worst score: {round(worst_score, 2)}")

#             print(f'best fitness: {round(best_score, 2)}') 
#             print(f'worst fitness: {round(worst_score, 2)}')
        
        # select two parents 
        p1_idx, p2_idx = tuple(np.random.choice(np.arange(len(pop)), size=2, replace=False)) 
        p1, p2 = self.get_route(pop[p1_idx]), self.get_route(pop[p2_idx])
        
        try: 
            # crossover then mutate 
            print('performing crossover & mutation')
            child = self.crossover([x['task group'] for x in p1], 
                                   [x['task group'] for x in p2])['route'] 
            child_route = self.order_task_list(child) 
            child_route = self.mutation(og_route=child_route, verbose=False)
            child_fit = self.fitness(child_route)  
            #print(child_fit)
        except Exception as e:  
            
            
            try:  
                if len(child) == 1: 
                    child_route = self.order_task_list([x['task group'] for x in child])  
                    child_fit = self.fitness(child_route)   
                else: 
                    print('len child is > 1 and some other error') 
                    print(child) 
                    print(e)
                    return np.nan
            except Exception as e:

                print(f'error... {e}')
                print('p1 ', p1) 
                print('p2 ', p2)
                return np.nan

        if child_fit['score'] > worst_score:  
            #print('child better than worst... replacing')
            # replace 
            pop[worst_idx] = child_route 
            fits[worst_idx] = child_fit
        
        return {'pop': pop, 'fits': fits}  
    
    
    def compute_drop_scores(self, route):  

        CONVERSION_FACTOR = 60
    
        "calculate drop scores for every node in a route" 

        # score every node in the route  

        start_node = 0 
        end_node = len(route) - 1 
        drop_scores = {}
        for i in range(len(route)):   

            node_score = route[i]['experimental_priority']
            node_time = route[i]['total time to complete'] * CONVERSION_FACTOR

            if i == start_node: 
                # previous is home 
                prev = self.start_loc  
                prev_task = self.start_loc 
                current = route[i]['site_code']  
                current_task = route[i]['task group']
                try: 
                    future = route[i+1]['site_code']  
                    future_task = route[i+1]['task group']
                except: 
                    print('no future node: ', route) 
                    future = self.end_loc
                    future_task = self.end_loc
            
            elif i == end_node: 
                # next is home  
                try:
                    prev = route[i-1]['site_code']    
                    prev_task = route[i-1]['task group']
                except: 
                    print('no prev node: ', route) 
                    prev = self.start_loc 
                    prev_task = self.start_loc 
                
                current = route[i]['site_code']
                current_task = route[i]['task group']
                future = self.end_loc 
                future_task = self.end_loc

            else: 
                # proceed as usual  
                prev = route[i-1]['site_code']
                current = route[i]['site_code']
                future = route[i+1]['site_code']  
                
                prev_task, current_task, future_task = route[i-1]['task group'], route[i]['task group'], route[i+1]['task group']

            # compute: drop(node) = score / (dist(prev, node) + dist(next, node) - dist(prev, next)) 

#            pc = self.dtable.query(f"start == '{prev}' and end == '{current}'")  
#            cf = self.dtable.query(f"start == '{current}' and end == '{future}'") 
#            pf = self.dtable.query(f"start == '{prev}' and end == '{future}'")   
            
            pc = self.get_distance(prev_task, current_task) 
            cf = self.get_distance(current_task, future_task) 
            pf = self.get_distance(prev_task, future_task) 

#            # need these i suppose 
#            if pc.empty: 
#                print(f'no distance found between {prev} and {current}')
#            else: 
#                pc = pc['value'].values[0]
#
#            if cf.empty: 
#                print(f'no distance found between {current} and {future}')
#            else: 
#                cf = cf['value'].values[0]
#
#            if pf.empty: 
#                print(f'no distance found between {prev} and {future}')
#            else: 
#                pf = pf['value'].values[0]

            # need these i suppose 
            if np.isnan(pc): 
                print(f'no distance found between {prev} and {current}')
  

            if np.isnan(cf): 
                print(f'no distance found between {current} and {future}')


            if np.isnan(pf): 
                print(f'no distance found between {prev} and {future}')


            drop_score = node_score / (pc + cf + node_time - pf)  
            drop_scores[i] = drop_score  
            

        return drop_scores 
    
    def drop_operator(self, route): 
        "drop nodes until tmax is satisfied"

        assert len(route) >= 2

        new_route = copy.deepcopy(route) 

        current_cost = self.fitness(new_route)['cost'] 

        while current_cost >= self.tmax:  
            # drop nodes selectively using drop scores 

            drop_scores = self.compute_drop_scores(new_route) 
            drop_idx = sorted(drop_scores.items(), key=lambda x: x[1])[0][0] 

            del new_route[drop_idx] 

            current_cost = self.fitness(new_route)['cost'] 

        return {'route': new_route, 'cost': current_cost} 
    
    def full_add_operator(self, route: list[dict], final=False): 
    
        """add operator used in mod case

        add until we can't anymore 
        
        TODO: if using max_sites, if current num transitions >= max_sites, filter 
        remaining nodes only to already visited sites 
        """   

        new_route = copy.deepcopy(route)

        current_cost = self.fitness(new_route)['cost']   
        #print(f'initial cost: {current_cost}')
        remaining_nodes = copy.deepcopy([x for x in self.task_list if x not in route]) 
        
        # ideally, this should allow us to only select tasks at facilities that have already been visited
        current_facilities = set([x['site_code'] for x in new_route])
        if len(current_facilities) >= self.MAX_SITES: 
            remaining_nodes = [x for x in remaining_nodes if x['site_code'] in current_facilities]
        
        if (current_cost > self.tmax) and (not final): 
            print('inputted route has a cost greater than tmax') 
            return new_route 
            
        if final: 
            # we've reached an optimal set of routes (using a time block), 
            # and want to only add tasks at visited facilities  
            # while it's feasible 
            
            fit = self.fitness(new_route)
            current_cost_modified = fit['cost'] - fit['transition_pen'] 
            
            if fit['transition_pen'] > 0: 
                current_cost = current_cost_modified # this is just current_cost less the transitions
                remaining_nodes = [x for x in remaining_nodes if x['site_code'] in current_facilities]

        assert current_cost < self.tmax 

        # while loop  
        while current_cost <= self.tmax: 
            # loop through all nodes not in route   
            best_add_score = 0  
            best_add_cost = np.inf 
            best_node_addition = None
            best_insertion_point = None
            for node in remaining_nodes: 
                add_out = self.add_operator(node_dict=node, route=new_route) 

                best = sorted(add_out, key=lambda x: x['feas']['value'], reverse=True)[0] 

                if best['feas']['value'] > best_add_score:  
                    best_add_cost = best['feas']['cost']
                    best_add_score = best['feas']['value']
                    best_node_addition = node 
                    best_insertion_point = best['insertion_point']


            # if the marginal cost + current cost <= tmax, store score, else score = 0 
            # select the node to add with maximum add value  
            if best_add_cost <= (self.tmax - current_cost):  

                best_node_addition['visited'] = True 
                
                if best_insertion_point[-1] == self.end_loc:  
                    new_route = new_route + [best_node_addition]  
                elif best_insertion_point[0] == self.start_loc: 
                    new_route = [best_node_addition] + new_route 

#                 # include best node in route  
#                 if self.name in best_insertion_point: 
#                     if best_insertion_point[-1] == self.name: # new node goes at very end 
#                         new_route = new_route + [best_node_addition] 
#                     else: # new node goes at the beg 
#                         new_route = [best_node_addition] + new_route 
                else:  
                    idx2 = new_route.index(best_insertion_point[1])  
                    new_route.insert(idx2, best_node_addition) 

                # update route fitness   
                if final: 
                    # total cost less transition pen 
                    fit = self.fitness(new_route)
                    current_cost = fit['cost'] - fit['transition_pen']  
                else:
                    current_cost = self.fitness(new_route)['cost'] 
                # print(f'new cost: {current_cost}') 

                # update remaining nodes 
                remaining_nodes.remove(best_node_addition)

            else: 
                break  


        return new_route 
    
    def tsp_variable_start_and_end(self, route, start, end): 

        """
        brute force implementation at site level – every permuation checked 

        allows for variable start and end locations 
        """ 

        sites = list(set([x['site_code'] for x in route]))
        site_perms = list(itertools.permutations(sites))  

        best_cost = np.inf
        best_route = None
        for perm in site_perms: 
            
            # format route w start and end  
            new_route = [start] + list(perm) + [end] 

            # eval  
            cost = 0 
            for a,b in pairwise(new_route): 
                # distance    
                row = self.dtable.query(f"start == '{a}' and end == '{b}'") 
                try:
                    cost += row['value'].values[0]
                except: 
                    return f"Error - could not find distance between {a} and {b}" 

            if cost < best_cost: 
                best_cost = cost 
                best_route = new_route  
                
        formatted = []
        for site in best_route[1:-1]: # 0 and -1 idx are start and end locs
            formatted += [x for x in route if x['site_code'] == site] 
            
        for s in formatted: 
            s['visited'] = True

        route_cost = self.fitness(formatted)  
        
        output = {'route': formatted, 'formatted_route': formatted, 
            'cost': {'travel': route_cost['travel_cost'], 
                     'time': route_cost['time_costs'], 
                     'total': route_cost['cost']}}
        
        return output
    
    def create_distance_matrix_og(self, route): 
    
        full_route = copy.deepcopy(route) 

        origin = (self.name, self.name)

        full_route = [origin] + [(x['task group'], x['site_code']) for x in full_route] + [origin] 
        cols = [x[0] for x in full_route][:-1]

        dist_matrix = pd.DataFrame(np.zeros((len(full_route)-1 , len(full_route)-1)), columns=cols, index=cols)

        for stop1 in full_route: 
            for stop2 in full_route:  
                if stop1[1] == stop2[1]:  
                    dist_matrix[stop1[0]][stop2[0]] = 0  
                else: 
                    # find distance  
                    row = self.dtable.query(f"start == '{stop1[1]}' and end == '{stop2[1]}'")  

                    if not row.empty: 
                        travel_time = row['value'].values[0]  
                    else: 
                        raise f"distance between {stop1[1]} and {stop2[1]} not found" 

                    dist_matrix[stop1[0]][stop2[0]] = travel_time 

        return dist_matrix.values, cols 



#    def create_distance_matrix(self, route):
#        full_route = copy.copy(route)
#        origin = (self.name, self.name)
#        full_route = [origin] + [(x['task group'], x['site_code']) for x in full_route] + [origin]
#        cols = [x[0] for x in full_route][:-1]
#        num_stops = len(full_route) - 1
#        dist_matrix = np.zeros((num_stops, num_stops))
#        stop1_codes = np.array([stop[1] for stop in full_route[:-1]])
#        stop2_codes = np.array([stop[1] for stop in full_route[:-1]])
#        i, j = np.meshgrid(stop1_codes, stop2_codes, indexing='ij')
#        mask = i == j
#        dist_matrix[mask] = 0
#        dist_matrix[~mask] = self.dtable.loc[(self.dtable['start'].values[:, None] == i[~mask]) & (self.dtable['end'].values[:, None] == j[~mask])]['value'].values
#        return dist_matrix, cols



    def solve_tsp(self, distance_matrix: np.array, indicies: list, tsp_solver='fast-tsp'):  
        
        def float_to_int(lst):
            return [[int(float) for float in inner] for inner in lst]
        
        CONVERSION_FACTOR = 60

        if tsp_solver == 'dp':
            r, cost = solve_tsp_dynamic_programming(distance_matrix) 
        elif tsp_solver == 'ls':
            r, cost = solve_tsp_local_search(distance_matrix, perturbation_scheme='two_opt')   
        elif tsp_solver == 'fast-tsp':  
            DL = float_to_int(distance_matrix)
            r = fast_tsp.greedy_nearest_neighbor(DL)
        else: 
            r, cost = solve_tsp_simulated_annealing(distance_matrix)

        formatted_route = [indicies[i] for i in r] + [self.end_loc] # doesn't include home node at end so we have to add it  
        
        route_obj = self.order_task_list(formatted_route) 
        route_cost = self.fitness(route_obj) 
        
        """
        {'score': task_scores, 'cost': total_cost, 
                'travel_cost': travel_costs, 'time_costs': task_costs}
        """
            
        return {'route': route_obj, 'formatted_route': formatted_route, 
            'cost': {'travel': route_cost['travel_cost'], 
                     'time': route_cost['time_costs'], 
                     'total': route_cost['cost']}} 
    
    
    def get_submatrix(self, indices): 
        
        """
        written by ChatGPT.. 
        
        filters down the full dmat 
        """ 
        
        indices_list = self.full_dmat_cols 
        distance_matrix = self.full_dmat
        
        # Create an empty list to store the indices of the rows and columns we want to keep
        keep_indices = []
        # Iterate through the indices passed to the function
        for index in indices:
            # Find the index of the current element in the indices_list
            i = indices_list.index(index)
            # Add the index to the list of indices to keep
            keep_indices.append(i)
        # Use the list of indices to keep to index the distance matrix and return a submatrix
        return distance_matrix[np.ix_(keep_indices, keep_indices)]  
    
    def solve_tsp_fixed_enpoints(self, route, start_loc, end_loc): 
    
        """
        thank god for this: https://stackoverflow.com/questions/14527815/how-to-fix-the-start-and-end-points-in-travelling-salesmen-problem 

        """ 
        
        # narrow it down to just sites to reduce search space 
        sites = list(set([x['site_code'] for x in route]))

        full_route = [start_loc] + sites + [end_loc]  

        start_col, end_col = f"start_{start_loc}", f"end_{end_loc}"

        cols = [start_col] + sites + [end_col]  

        col_stop_mapping = {s:c for s,c in zip(full_route, cols)}

        dist_matrix = pd.DataFrame(np.zeros((len(full_route)+1 , len(full_route)+1)), 
                                   columns=cols+['dummy'], index=cols+['dummy'])

        s1_counter = 0 
        for stop1 in full_route:  
            s1_col = cols[s1_counter] 

            s2_counter = 0
            for stop2 in full_route:    
                s2_col = cols[s2_counter] 
                if stop1 == stop2:  
                    dist_matrix[s1_col][s2_col] = 0   
                else: 
                    # find distance  
                    row = self.dtable.query(f"start == '{stop1}' and end == '{stop2}'")  

                    if not row.empty: 
                        travel_time = row['value'].values[0]  
                    else: 
                        print( f"distance between {stop1} and {stop2} not found" ) 
                        return np.nan

                    dist_matrix[s1_col][s2_col] = travel_time   

                s2_counter += 1
            s1_counter += 1


        # do the hack -- adding dummy node with dist = inf everywhere EXCEPT desired start / end nodes 
        dummy = [] 
        for c in cols: 
            if c in [start_col, end_col]: 
                dummy.append(0)  
            else: 
                dummy.append(np.inf)  

        dummy += [0] # needs a 0 dist to itself 

        dist_matrix['dummy'] = dummy
        dist_matrix.loc['dummy'] = dummy

        # solve it w/ dynamic programming to ensure exactness 
        r, _ = solve_tsp_dynamic_programming(dist_matrix.values)   
        full_cols = cols + ['dummy']

        path = [full_cols[i] for i in r] 

        # if dummy is at beginning of route, reverse it, else keep 
        if 'dummy' in path[:2]:  
            path.reverse()  


        # get rid of vars we don't need 
        path.remove(start_col) 
        path.remove(end_col) 
        path.remove('dummy')  

        formatted = []
        for site in path:
            formatted += [x for x in route if x['site_code'] == site] 

        for s in formatted: 
            s['visited'] = True


        route_cost = self.fitness(formatted)  

        output = {'route': formatted, 'formatted_route': formatted, 
            'cost': {'travel': route_cost['travel_cost'], 
                     'time': route_cost['time_costs'], 
                     'total': route_cost['cost']}}


        return output

    def tour_improvement(self, route):  
        
        "wrap the tsp functions together"   
        dmat_cols = [x['task group'] for x in route]
        
        if self.custom_locs: 
            # solve w brute force method 
            # tsp_solution = self.tsp_variable_start_and_end(route, start=self.start_loc, end=self.end_loc) 
            tsp_solution = self.solve_tsp_fixed_enpoints(route, self.start_loc, self.end_loc)
        else: 
            #distance_matrix, cols = self.create_distance_matrix(route)   
            t = time.time()
            distance_matrix = self.get_submatrix(dmat_cols) 
            tt = time.time() 
            print(f"submatrix calc took {round(tt - t, 2)} sec") 
            
            t1 = time.time()
            tsp_solution = self.solve_tsp(distance_matrix, dmat_cols, tsp_solver='fast-tsp') 
            t2 = time.time() 
            print(f"tsp calc took {round(t2 - t1, 2)} sec") 
            
        return tsp_solution

    
    def mod_gen(self, pop):  
        
        """
        implement the mode == 0 case here 
        
        returns a new population 
        """ 
        
        tsp_times, drop_times, add_times = [], [], []
        
        # 1. tsp  
        new_pop = [] 
        i = 0 
        for route in pop:  
            print(i)
            s = time.time()
            tsp = self.tour_improvement(route)   
            e = time.time() 
            tsp_times.append(e - s) 
            
            new_route = tsp['route']
            
            
            # 2. if cost > tmax -> drop operator, else add operator  
            if tsp['cost']['total'] > self.tmax: 
                # drop operator   
                s = time.time()
                drop_res = self.drop_operator(new_route)  
                e = time.time() 
                drop_times.append(e - s)
                revised_route = drop_res['route'] 
                
            else: 
                # add operator   
                s = time.time()
                add_res = self.full_add_operator(new_route)  
                e = time.time() 
                add_times.append(e - s)
                revised_route = add_res 
                
            new_pop.append(revised_route) 
            i += 1
        
        return new_pop, {'tsp': tsp_times, 'drops': drop_times, 'adds': add_times} 
    
    def parse_ga_output(self): 
        
        """
        first modify current fits to work with facility time blocks
        """ 
        
        if 1 == 2: # self.custom_locs
            current_pop_modified = self.current_pop 
            current_fits_modified = self.current_fits 
        else:
            current_pop_modified = [self.full_add_operator(r, final=True) for r in self.current_pop if len(r) > 0]  
            current_fits_modified = [self.fitness(r) for r in current_pop_modified]
    
        enum_fits = sorted(enumerate(current_fits_modified), key=lambda x: x[1]['score'], reverse=True) 
        best_idx = enum_fits[0][0] 
    
        # make sure this is tsp'd
        if not self.custom_locs:
            route_full = self.tour_improvement(current_pop_modified[best_idx])['route'] 
        else: 
            route_full = current_pop_modified[best_idx] 
            
        route_task_groups = [x['task group'] for x in route_full] 
        route_sites = [x['site_code'] for x in route_full]   
        
        route_fitness = self.fitness(route_full, output=True) 
        score = round(route_fitness['score'], 2) 
        cost_hrs = round(route_fitness['cost'] / 60**2, 2)  
        travel_hrs = round(route_fitness['travel_cost'] / 60**2, 2) 
        block_hrs = round(route_fitness['transition_pen'] / 60**2, 2)
        task_hrs = round(route_fitness['time_costs'] / 60**2, 2)
        #time_on_road_pct = round(route_fitness['travel_cost'] / enum_fits[0][1]['cost'], 2 )  
        travel_times = route_fitness['travel_times'] 
        cost_less_transition_pen = round(cost_hrs - block_hrs, 2)
        time_on_road_pct = round(travel_hrs / cost_less_transition_pen, 2)
        
        univisited = [x for x in self.task_list if x['task group'] not in route_task_groups]

        return { 
            'fitness': {'score': score, 'total_cost': cost_hrs, 'cost_less_transition_pen': cost_less_transition_pen,
                        'travel_cost': travel_hrs, 'task_hrs': task_hrs, 
                        'time_on_road': time_on_road_pct, 'block_hrs': block_hrs, 'travel_times':travel_times
                       }, 
            'route_full': route_full, 
            'route_task_groups': route_task_groups, 
            'route_sites': route_sites, 
            'unvisited': univisited
        }
    
    def run(self, d2d, iters=10): 
        
        pop = self.create_initial_pop()  
        self.current_pop = pop
        pop_fitness = [self.fitness(self.get_route(p)) for p in pop]    
        self.current_fits = pop_fitness
        
        self.reg_gen(pop, pop_fitness) 
        
        self.fitness_tracker = []
        
        print('')
        i = 0 
        while i <= iters:  
            print(f'generation {i}') 
            
            if i % d2d == 0:  
                print('**mod generation**')
                pop, time_test = self.mod_gen(self.current_pop)  
                pop_fitness = [self.fitness(r) for r in pop]  
                self.gen = 'mod' 
                self.runtimes.append(time_test)
            else:
                reg_gen = self.reg_gen(self.current_pop, self.current_fits)    
                pop, pop_fitness = reg_gen['pop'], reg_gen['fits']
                self.gen = 'regular' 
                
            
            self.current_pop = pop
            self.current_fits = pop_fitness 
            self.fitness_tracker.append(pop_fitness)
            print('')
            i += 1 
        
        print('making final route optimizations... almost done')
        self.output = self.parse_ga_output() 
        
        return self.output 
    
    
def schedule_appt(appt_site, 
              time_into_day, 
              total_hours,  
              task_list, 
              name, 
              dtable, 
              appt_length=15
             ):  
    
    """
    build schedule for before and after appt  
    
    time_into_day: SECONDS 
    """  
    
    print(f"--- Generating schedule at {appt_site} for {time_into_day} into day --- ")
    
    # check if travel time is close to time into day 
    # if so, just travel there, if not we can route  
    
    THRESHOLD = .16666*60*60  # 10 minutes // can be changed 
    APPT_TIME = appt_length  
    APPT_SCORE = 0
    
    first_route_done, second_route_done = False, False
    
    drow = dtable.query(f"start == '{name}' and end == '{appt_site}'")  
    
    assert (not drow.empty) 
    
    if not drow.empty: 
        dist = drow['value'].values[0] 
        if abs(time_into_day - dist) < THRESHOLD: 
            # route to site not gonna work  
            first_route = [
            {'task group': f'Appointment -- {appt_site}',
              'number of tasks': 1,
              'total time to complete': APPT_TIME,
              'experimental_priority': APPT_SCORE,
              'site_code': appt_site,
              'visited': True}
            ]    
            
            cost = drow['value'].values[0] + (APPT_TIME*60)
            
            fitness1 = {'score': APPT_SCORE, 
                        'travel_cost': round(drow['value'].values[0] / 60**2, 2), 
                        'cost_less_transition_pen': round(cost / 60**2, 2), 
                       'travel_times': [round(drow['value'].values[0] / 60, 2)]}
            
            print('site too far from home, first half of day directly to site')
            
            first_route_done = True 
            
        if abs((total_hours - time_into_day) - dist) < THRESHOLD:  
            # need to just go straight home 
            second_route = [
            {'task group': f'Appointment -- {appt_site}',
              'number of tasks': 1,
              'total time to complete': APPT_TIME,
              'experimental_priority': APPT_SCORE,
              'site_code': appt_site,
              'visited': True}
            ]   
            
            cost = drow['value'].values[0] + (APPT_TIME*60)
            
            fitness2 = {'score': APPT_SCORE, 
                        'travel_cost': round(drow['value'].values[0] / 60**2, 2), 
                        'cost_less_transition_pen': round(cost / 60**2, 2), 
                       'travel_times': [round(drow['value'].values[0] / 60, 2)]}
            
            print('site too far from home, second half of day directly to site')
            
            second_route_done = True
    
    if not first_route_done: 
        print('generating first route') 
        ga = GA(task_list=task_list, 
                     name=name, 
                     dtable=dtable, 
                     tmax=time_into_day, 
                     pop_size=10, 
                     block_time=0*60*60, max_sites=None, 
                    start_end_locs={'start': None, 'end': appt_site, 
                                    'start_is_rd': False, 'end_is_rd': True}
           )   
        try:
            ga.run(d2d=20, iters=60)  
        
            first_route = ga.output['route_full'] + [{'task group': f'Appointment -- {appt_site}',
                                                  'number of tasks': 1,
                                                  'total time to complete': APPT_TIME,
                                                  'experimental_priority': 0,
                                                  'site_code': appt_site,
                                                  'visited': True}] 
        
            fitness1 = ga.output['fitness'] 
        except Exception as e: 
            print('could not generate first route: ', e) 
            first_route = [
            {'task group': f'Appointment -- {appt_site}',
              'number of tasks': 1,
              'total time to complete': APPT_TIME,
              'experimental_priority': APPT_SCORE,
              'site_code': appt_site,
              'visited': True}
            ]
            
            cost = drow['value'].values[0] + (APPT_TIME*60) 
            
            fitness1 = {'score': APPT_SCORE, 
                        'travel_cost': round(drow['value'].values[0] / 60**2, 2), 
                        'cost_less_transition_pen': round(cost / 60**2, 2), 
                       'travel_times': [round(drow['value'].values[0] / 60, 2)]}
        
        
    if not second_route_done:   
        
        print('generating second route')
        
        # remove task groups in first route 
        visited = [x['task group'] for x in first_route]
        revised_task_list = [x for x in task_list if x['task group'] not in visited]  
        
        remaining_hours = (total_hours) - time_into_day - (appt_length*60)   
        
        print(f"{remaining_hours=} | {total_hours=} | {time_into_day=} | appt_length={(appt_length*60)}")
        
        try:   
            print({'start': appt_site, 'end': None, 'start_is_rd': True, 'end_is_rd': False})
            
            ga = GA(task_list=revised_task_list, 
                     name=name, 
                     dtable=dtable, 
                     tmax=remaining_hours, 
                     pop_size=10, 
                     block_time=0*60*60, max_sites=None, 
                    start_end_locs={'start': appt_site, 'end': None, 
                                    'start_is_rd': True, 'end_is_rd': False}
            ) 
            ga.run(d2d=20, iters=60)  

            second_route = ga.output['route_full']    
            fitness2 = ga.output['fitness']
            
        except Exception as e: 
            print('could not generate second route: ', e)  
            
            second_route = [
                {'task group': f'Appointment -- {appt_site}',
                  'number of tasks': 1,
                  'total time to complete': APPT_TIME,
                  'experimental_priority': APPT_SCORE,
                  'site_code': appt_site,
                  'visited': True}
                ]   
            
            cost = drow['value'].values[0] + (APPT_TIME*60)
            
            fitness2 = {'score': APPT_SCORE, 
                        'travel_cost': round(drow['value'].values[0] / 60**2, 2), 
                        'cost_less_transition_pen': round(cost / 60**2, 2), 
                       'travel_times': [round(drow['value'].values[0] / 60, 2)]}
            
    full = first_route + second_route  
    fit = {
        'score': fitness1['score'] + fitness2['score'], 
        'total_cost': fitness1['cost_less_transition_pen'] + fitness2['cost_less_transition_pen'], 
        'cost_less_transition_pen': fitness1['cost_less_transition_pen'] + fitness2['cost_less_transition_pen'],
        'travel_cost': fitness1['travel_cost'] + fitness2['travel_cost'], 
        'task_hrs': (fitness1['cost_less_transition_pen'] + fitness2['cost_less_transition_pen']) - (fitness1['travel_cost'] + fitness2['travel_cost']), 
        'time_on_road': round((fitness1['travel_cost'] + fitness2['travel_cost']) / (fitness1['cost_less_transition_pen'] + fitness2['cost_less_transition_pen']), 2)
    } 
    
    visited_tg = [x['task group'] for x in full]
    unvisited = [x for x in task_list if x['task group'] not in visited_tg] 
    
    new_full = []  
    v = []
    for stop in full: 
        if stop['task group'] not in v: 
            new_full.append(stop) 
            v.append(stop['task group']) 
            
    ttimes = fitness1['travel_times'] + fitness2['travel_times']  
    fit['travel_times'] = ttimes
    #ttimes = ttimes[:len(new_full)+1]
    
    return {
        'reoptimized_route': True, 
        'reoptimized_timestamp': str(np.datetime64('now')),  
        'revision': {'type': 'appointment', 'value': appt_site}, 
        'reoptimized_summary': { 
                                'appt_time_into_day': time_into_day, 
                                'first_route': first_route, 
                                'second_route': second_route, 
                                'fitness_breakdown': {'first': fitness1, 'second': fitness2}
                               }, 
        'route_full': new_full, # TODO: check for dups   
        'route_sites': [x['site_code'] for x in new_full],   
        'route_task_groups': [x['task group'] for x in new_full], 
        'fitness': fit, 
        #'travel_times': ttimes, 
        'unvisited': unvisited
    }
        