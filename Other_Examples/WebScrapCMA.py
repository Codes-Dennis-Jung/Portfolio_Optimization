import requests
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_blackrock_assumptions():
    """Download and process BlackRock assumptions data"""
    url = "https://www.blackrock.com/blk-inst-c-assets/images/tools/blackrock-investment-institute/cma/blackrock-capital-market-assumptions.xlsx"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        logger.info(f"Downloading file from {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        df = pd.read_excel(
            response.content,
            sheet_name='Assumptions',
            header=[1,2],
            index_col=[0, 1, 2, 3]
        )
        
        df = df[df.index.get_level_values(0) == 'EUR']
        
        columns_to_keep = df.columns[
            (df.columns.get_level_values(0).str.contains('Expected returns', na=False) & 
             df.columns.get_level_values(1).str.contains('10 year', na=False)) |
            (df.columns.get_level_values(0).str.contains('Lower mean uncertainty', na=False) & 
             df.columns.get_level_values(1).str.contains('10 year', na=False)) |
            (df.columns.get_level_values(0).str.contains('Upper mean uncertainty', na=False) & 
             df.columns.get_level_values(1).str.contains('10 year', na=False)) |
            df.columns.get_level_values(0).str.contains('Volatility', na=False)
        ]
        df = df[columns_to_keep]
        
        return clean_assumptions_data(df)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

def clean_assumptions_data(df):
    """Clean and process the assumptions data"""
    try:
        df = df.copy()
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        df.columns = [f"{col[0]}_{col[1]}".strip() if pd.notna(col[1]) else col[0].strip() 
                     for col in df.columns]
        
        df = df.reset_index()
        df = df.drop('level_0', axis=1)
        
        new_column_names = {
            'level_1': 'Asset Class',
            'level_2': 'Market',
            'level_3': 'Index'
        }
        
        for col in df.columns:
            if 'Expected returns' in col and '10 year' in col:
                new_column_names[col] = 'Expected returns'
            elif 'Lower mean uncertainty' in col and '10 year' in col:
                new_column_names[col] = 'Lower mean uncertainty'
            elif 'Upper mean uncertainty' in col and '10 year' in col:
                new_column_names[col] = 'Upper mean uncertainty'
            elif 'Volatility' in col:
                new_column_names[col] = 'Volatility'
        
        df = df.rename(columns=new_column_names)
        
        numeric_columns = ['Expected returns', 'Lower mean uncertainty', 
                         'Upper mean uncertainty', 'Volatility']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')   # Convert to decimal
        
        df = df.dropna(subset=numeric_columns)  # Remove rows with missing numeric values
        
        ordered_cols = ['Asset Class', 'Market', 'Index', 'Expected returns',
                       'Lower mean uncertainty', 'Upper mean uncertainty', 'Volatility']
        df = df[ordered_cols]
        
        return df
    
    except:
        pass

# Get BlackRock data
exp_ret = download_blackrock_assumptions()
exp_ret = exp_ret.set_index(['Asset Class', 'Market', 'Index'])