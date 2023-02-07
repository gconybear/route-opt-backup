import requests 
import json
import streamlit as st 

def extract_lat_long_via_address(address_or_zipcode, rtype='coords'):
    lat, lng = None, None
    api_key = st.secrets['GEOCODE_KEY']
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    endpoint = f"{base_url}?address={address_or_zipcode}&key={api_key}"
    # see how our endpoint includes our API key? Yes this is yet another reason to restrict the key
    r = requests.get(endpoint) 
    
    try:
        if r.status_code == 200: 
            res = json.loads(r.content)  
            if rtype == 'coords':  
                res_obj = res['results'][0]['geometry']['location']  
                res_obj.update({'formatted_address': res['results'][0]['formatted_address']}) 
                
                return res_obj
            else: 
                return res
    except: 
        return None 
    
    return None