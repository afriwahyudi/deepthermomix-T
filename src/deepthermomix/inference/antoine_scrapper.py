from rdkit import Chem
import requests
from bs4 import BeautifulSoup
import re

class AntoineEquation:
    """
    Calculates Saturation Pressure (P_sat) in Bar.
    Equation: log10(P_sat) = A - (B / (T + C))
    
    Scrape NIST WebBook by:
      1. Attempting InChI search (precise).
      2. Attempting Name search (fallback).
      3. Extracting the official Name from the NIST page for plotting.
    """
    def __init__(self):
        self.cache = {}
        self.names = {}
        self.base_url = "https://webbook.nist.gov/cgi/cbook.cgi"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        })

    def _get_inchi(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToInchi(mol)
        return None
    
    def _canonicalize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return smiles

    def _fetch_nist_params(self, smiles, name=None):
        """Scrapes NIST WebBook for Antoine parameters and Name."""
        inchi = self._get_inchi(smiles)
        entries = []
        scraped_name = None
        
        if inchi:
            print(f"Attempting NIST InChI Search for: {smiles}")
            entries, scraped_name = self._query_nist({'InChI': inchi, 'Units': 'SI', 'Mask': '4'})
        
        if not entries and name:
            print(f"InChI search empty. Retrying with Name: {name}")
            entries, scraped_name = self._query_nist({'Name': name, 'Units': 'SI', 'Mask': '4'})

        if entries:
            print(f"  -> Success: Found {len(entries)} parameter sets.")
            if scraped_name:
                print(f"  -> Identified as: {scraped_name}")
                self.names[smiles] = scraped_name
        else:
            print(f"  -> Failed: No parameters found for {smiles} (Name: {name}).")
            
        return entries

    def _query_nist(self, params):
        entries = []
        scraped_name = None
        try:
            req = requests.Request('GET', self.base_url, params=params)
            prepped = self.session.prepare_request(req)
            print(f"  -> Querying: {prepped.url}")
            
            response = self.session.send(prepped, timeout=15)
            
            if response.status_code != 200:
                print(f"  -> HTTP Error {response.status_code}")
                return [], None
                
            content = response.content
        except Exception as e:
            print(f"  -> Network error: {e}")
            return [], None

        soup = BeautifulSoup(content, 'html.parser')
        header = soup.find('h1', id='Top')
        if header:
            scraped_name = header.get_text().strip()

        tables = soup.find_all('table')
        target_tables = []
        
        for t in tables:
            if t.get('aria-label') == 'Antoine Equation Parameters':
                target_tables.append(t)
            elif t.find_previous_sibling('h3') and 'Antoine' in t.find_previous_sibling('h3').text:
                 target_tables.append(t)

        for table in target_tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    try:
                        temp_text = cols[0].get_text().strip()
                        temp_matches = re.findall(r"[-+]?\d*\.\d+|\d+", temp_text)
                        
                        if len(temp_matches) >= 2:
                            t_min, t_max = float(temp_matches[0]), float(temp_matches[1])
                        elif len(temp_matches) == 1:
                            val = float(temp_matches[0])
                            t_min, t_max = val - 10, val + 10 
                        else:
                            t_min, t_max = 0.0, 5000.0 

                        val_A = float(cols[1].get_text().replace(' ',''))
                        val_B = float(cols[2].get_text().replace(' ',''))
                        val_C = float(cols[3].get_text().replace(' ',''))
                        
                        entries.append({
                            'A': val_A, 'B': val_B, 'C': val_C,
                            't_min': t_min, 't_max': t_max
                        })
                    except (ValueError, IndexError):
                        continue
        return entries, scraped_name

    def get_Psat(self, smiles, T_kelvin, name=None):
        if smiles in self.cache:
            params_list = self.cache[smiles]
        else:
            params_list = self._fetch_nist_params(smiles, name)
            self.cache[smiles] = params_list
        
        if not params_list:
            print(f"Warning: No Antoine params found for {smiles}. VLE will be inaccurate (Using P=1.0 bar).")
            return 1.0
        
        best_params = None
        
        # Priority 1: T is strictly within range
        for p in params_list:
            if p['t_min'] <= T_kelvin <= p['t_max']:
                best_params = p
                break
        
        # Priority 2: Closest range (Extrapolation)
        if best_params is None:
            def dist_to_range(p):
                if T_kelvin < p['t_min']: return p['t_min'] - T_kelvin
                if T_kelvin > p['t_max']: return T_kelvin - p['t_max']
                return 0
            best_params = min(params_list, key=dist_to_range)
            print(f"  -> Note: T={T_kelvin}K is outside NIST range ({best_params['t_min']}-{best_params['t_max']}). Extrapolating.")

        A, B, C = best_params['A'], best_params['B'], best_params['C']
        log_p = A - (B / (T_kelvin + C))
        return 10**log_p
    
    def get_stored_name(self, smiles):
        """Retrieve name scraped from NIST if available."""
        return self.names.get(smiles, None)

