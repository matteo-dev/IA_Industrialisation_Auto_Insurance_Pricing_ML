# Python file containing JSON payload processing functions for the auto pricing API.

# Import required libraries
import pandas as pd
import numpy as np

# Function to process a single JSON payload
def process_single_payload(payload: dict) -> pd.DataFrame:
    """Transforms a single payload dictionary (JSON) into an enriched DataFrame."""
    df = pd.DataFrame([payload])
    
    # Replace outliers / data entry errors
    for col in ['anciennete_permis1', 'anciennete_permis2']:
        if col in df.columns:
            df[col] = df[col].replace(111, np.nan)
            
    # Geographic feature extraction
    if 'code_postal' in df.columns:
        df['departement'] = df['code_postal'].astype(str).str.zfill(5).str[:2]
        
    # Interaction features (handling potential missing/null values)
    age1 = df.get('age_conducteur1', pd.Series([0]))
    permis1 = df.get('anciennete_permis1', pd.Series([0]))
    df['age_obtention_permis1'] = age1 - permis1
    df['is_jeune_permis'] = (permis1 < 3).astype(int)
    
    din = df.get('din_vehicule', pd.Series([0]))
    poids = df.get('poids_vehicule', pd.Series([0]))
    df['ratio_puissance_poids'] = din / (poids + 1)
    
    prix = df.get('prix_vehicule', pd.Series([0]))
    
    # Specific handling for anciennete_vehicule which can be null (None)
    anc_veh = df.get('anciennete_vehicule', pd.Series([0])).fillna(0)
    df['ratio_valeur_anciennete'] = prix / (anc_veh + 1)
    
    age2 = df.get('age_conducteur2', pd.Series([0]))
    df['conducteur2_jeune'] = ((age2 > 0) & (age2 < 25)).astype(int)
    
    return df