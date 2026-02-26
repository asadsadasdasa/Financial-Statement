import pandas as pd
import streamlit as st
import os
import re
import csv
import io
import codecs
import unicodedata
import re
from utils import meses_fiscais

def find_file(keyword, excludes=None, max_up=3):
    """Locate a CSV whose filename contains `keyword`.

    - Searches the current working directory and up to `max_up` parent
      directories.
    - Falls back to a recursive os.walk from the current directory.
    - **Automatically ignores** filenames that are TEMPLATES (e.g. ending with 
      '_template.csv' or '_template_*.csv') to avoid picking up template files  
      when looking for raw data. Additional exclusions may be passed via the 
      ``excludes`` argument, which is a list of substrings to skip.
    - Returns the absolute path of the first matching file, or ``None`` if
      nothing is found.
    """
    # Only exclude templates, not files due to common words like 'model' or 'flat'
    default_excludes = ['_template', 'template_'] if excludes is None else []
    if excludes is None:
        excludes = default_excludes
    else:
        excludes = default_excludes + excludes
    
    cwd = os.getcwd()
    # construir lista de diretórios para procurar: cwd, parent, parent2, ...
    search_paths = [cwd]
    cur = cwd
    for _ in range(max_up):
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        search_paths.append(parent)
        cur = parent

    for path in search_paths:
        try:
            for f in os.listdir(path):
                if keyword.lower() in f.lower() and f.lower().endswith('.csv'):
                    if any(exc.lower() in f.lower() for exc in excludes):
                        continue
                    full = os.path.join(path, f)
                    try:
                        st.info(f"Matched data file for '{keyword}': {full}")
                    except Exception:
                        pass
                    return full
        except Exception:
            # permissão ou path inválido; ignora e continua
            continue

    # fallback: procurar recursivamente a partir do cwd
    for root, dirs, files in os.walk(cwd):
        for f in files:
            if keyword.lower() in f.lower() and f.lower().endswith('.csv'):
                if any(exc.lower() in f.lower() for exc in excludes):
                    continue
                full = os.path.join(root, f)
                try:
                    st.info(f"Found data file via walk for '{keyword}': {full}")
                except Exception:
                    pass
                return full

    return None


def detect_sep_and_encoding(path, sample_bytes=2048):
    """Tenta detectar separador e encoding do CSV lendo uma amostra.
    Retorna tupla (sep, encoding)."""
    # Primeiro tentar utf-8
    for enc in ('utf-8', 'utf-8-sig', 'latin1'):
        try:
            with open(path, 'r', encoding=enc, errors='strict') as f:
                sample = f.read(sample_bytes)
            # heurística simples: se houver mais tabs do que vírgulas ou ponto-e-vírgulas, presumimos tab
            tab_count = sample.count('\t')
            comma_count = sample.count(',')
            semicolon_count = sample.count(';')
            if tab_count > max(comma_count, semicolon_count) and tab_count > 0:
                return '\t', enc
            # Uso simples do csv.Sniffer para tentar detectar delimiter
            try:
                dialect = csv.Sniffer().sniff(sample)
                sep = dialect.delimiter
            except Exception:
                # heurística: tab, ;, ,
                if '\t' in sample:
                    sep = '\t'
                elif ';' in sample:
                    sep = ';'
                else:
                    sep = ','
            return sep, enc
        except Exception:
            continue
    return ',', 'latin1'


def find_header_row(path, max_lines=200):
    """Tenta localizar a linha que contém os cabeçalhos reais em exports SAP.
    Retorna o número de linhas a pular (skiprows) para que pandas leia corretamente o header.
    """
    sep, enc = detect_sep_and_encoding(path)
    tokens = ['dt', 'dtlcto', 'dt lcto', 'dt.lcto', 'conta', 'montante', 'mont', 'cen', 'cen lucro', 'cen.lucro', 'empr', 'moed']
    try:
        with open(path, 'r', encoding=enc, errors='replace') as f:
            for idx in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lnorm = unicodedata.normalize('NFKD', line).encode('ascii', 'ignore').decode('ascii').lower()
                # quick token count
                count = sum(1 for t in tokens if t in lnorm)
                if count >= 2:
                    return idx
    except Exception:
        return None
    return None


def read_csv_smart(path, **kwargs):
    """Lê CSV tentando detectar separador e encoding automaticamente."""
    sep, enc = detect_sep_and_encoding(path)
    try:
        df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False, **kwargs)
    except UnicodeDecodeError:
        # fallback direto para latin1 se utf-8 falhar
        try:
            df = pd.read_csv(path, encoding='latin1', sep=sep, low_memory=False, **kwargs)
        except Exception:
            df = pd.read_csv(path, encoding='latin1', low_memory=False, **kwargs)
    except Exception:
        # última tentativa sem especificar sep (pandas tenta adivinhar)
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin1', low_memory=False, **kwargs)

    # if we got a single-column result and the header string contains other typical delimiters,
    # try again with a more obvious separator
    if df.shape[1] == 1 and isinstance(df.columns[0], str):
        col0 = df.columns[0]
        if '\t' in col0 or '\t' in path:
            try:
                df = pd.read_csv(path, encoding=enc, sep='\t', low_memory=False, **kwargs)
            except Exception:
                pass
        elif ';' in col0:
            try:
                df = pd.read_csv(path, encoding=enc, sep=';', low_memory=False, **kwargs)
            except Exception:
                pass
    return df


def validate_columns(df, expected_cols, name):
    if df is None or df.empty:
        return []
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        try:
            st.warning(f"{name}: colunas esperadas ausentes: {missing}")
        except Exception:
            pass
    return missing

def clean_financial_number(x):
    if pd.isna(x): return 0.0
    s = str(x).strip()
    # Se o número tiver ponto E vírgula (Ex: 1.234,56 ou 1,234.56)
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'): # Padrão BR/EUR
            s = s.replace('.', '').replace(',', '.')
        else: # Padrão US
            s = s.replace(',', '')
    # Se tiver só vírgula
    elif ',' in s:
        if len(s.split(',')[-1]) == 2: s = s.replace(',', '.') # 1234,56
        else: s = s.replace(',', '') # 1,234
    try: return float(s)
    except: return 0.0

def apply_fiscal_calendar(df, date_col='Posting Date'):
    if not df.empty and date_col in df.columns:
        df['Posting Date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False)
        df['Cal_Ano'] = df['Posting Date'].dt.strftime('%Y')
        df['Cal_Mes'] = df['Posting Date'].dt.strftime('%m')
        df['Fiscal Year'] = df['Posting Date'].apply(lambda x: f"FY{str(x.year)[-2:]}/{str(x.year+1)[-2:]}" if pd.notnull(x) and x.month >= 4 else f"FY{str(x.year-1)[-2:]}/{str(x.year)[-2:]}" if pd.notnull(x) else "Unknown")
        df['Fiscal Period'] = df['Posting Date'].dt.month.map(meses_fiscais)
    return df

def classify_pl_line(pl_lvl_5, account):
    """Classify a P&L line item and determine its sign convention.
    
    SAP FAGLL03 exports use accounting signs:
    - Revenues (7xx accounts, Net Sales) come in NEGATIVE: must multiply by -1
    - Costs (6xx accounts, Operating Expenses) come in POSITIVE: keep as negative (or negate)
    
    Returns:
        str: One of 'REVENUE', 'COST', or 'OTHER'
    """
    if pd.isna(pl_lvl_5):
        if pd.isna(account):
            return 'OTHER'
        acc_str = str(account).strip()
        if acc_str.startswith('7'):
            return 'REVENUE'
        elif acc_str.startswith('6'):
            return 'COST'
        return 'OTHER'
    
    pl_str = str(pl_lvl_5).upper().strip()
    if 'NET SALES' in pl_str or 'REVENUE' in pl_str:
        return 'REVENUE'
    elif 'COST' in pl_str or 'EXPENSE' in pl_str or 'OPERATING' in pl_str or 'DEPRECIATION' in pl_str or 'AMORT' in pl_str:
        return 'COST'
    else:
        # fallback: check account prefix
        if not pd.isna(account):
            acc_str = str(account).strip()
            if acc_str.startswith('7'):
                return 'REVENUE'
            elif acc_str.startswith('6'):
                return 'COST'
    return 'OTHER'

def normalize_sap_signs(df, amount_col='Amount in local currency', pl_col='P&L LVL 5', account_col='Account'):
    """Normalize SAP accounting signs to FP&A conventions.
    
    - Revenues must be POSITIVE (multiply by -1 if SAP exported them as negative)
    - Costs must be NEGATIVE (so Gross Margin = Revenue + Costs works mathematically)
    - Creates 'Amount Normalized' columns with corrected signs
    - For dual-currency data (Actuals), normalizes both Montante em moeda interna AND Montante em MI2
    
    Args:
        df: DataFrame with SAP data
        amount_col: Column name containing amounts (default: 'Amount in local currency')
        pl_col: Column name for P&L classification (default: 'P&L LVL 5')
        account_col: Column name for account numbers (default: 'Account')
    
    Returns:
        DataFrame with 'Amount Normalized' column(s)
        For dual-currency Actuals, creates both normalized amount columns
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    def normalize_row(val, pl_type):
        """Apply sign normalization based on P&L line type"""
        if pd.isna(val) or val == 0:
            return 0.0
        
        val = float(val)
        
        # SAP convention: revenues are negative, costs are positive
        # FP&A convention: revenues are positive, costs are negative
        if pl_type == 'REVENUE':
            return abs(val)  # Make positive
        elif pl_type == 'COST':
            return -abs(val)  # Make negative
        else:
            return val  # Leave as-is
    
    # Classify each row once
    if pl_col in df.columns and account_col in df.columns:
        df['_pl_type'] = df.apply(
            lambda row: classify_pl_line(row.get(pl_col), row.get(account_col)),
            axis=1
        )
    else:
        df['_pl_type'] = 'OTHER'
    
    # Normalize the primary amount column
    if amount_col in df.columns:
        df['Amount Normalized'] = df.apply(
            lambda row: normalize_row(row[amount_col], row['_pl_type']),
            axis=1
        )
    
    # Also normalize Montante em MI2 if present (dual-currency Actuals)
    if 'Montante em MI2' in df.columns and amount_col != 'Montante em MI2':
        df['Montante em MI2 Normalized'] = df.apply(
            lambda row: normalize_row(row['Montante em MI2'], row['_pl_type']),
            axis=1
        )
    
    # Clean up temporary column
    df = df.drop(columns=['_pl_type'], errors='ignore')
    
    return df

@st.cache_data
def load_master_data():
    df_fagl, df_fc, df_cap, df_bud, df_curr = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 1. ACTUALS (Agora lê tanto o raw do SAP quanto o arquivo pré-limpo)
    actuals_file = find_file('actual base') or find_file('fagll03')
    if actuals_file:
        try:
            # tentar detectar a linha de header (útil em exports SAP com texto acima do header)
            header_row = find_header_row(actuals_file)
            if header_row is not None and header_row > 0:
                try: st.info(f"Detected header at line {header_row+1} in {actuals_file}")
                except Exception: pass
                df_fagl = read_csv_smart(actuals_file, on_bad_lines='skip', skiprows=header_row)
            else:
                df_fagl = read_csv_smart(actuals_file, on_bad_lines='skip')
            df_fagl.columns = df_fagl.columns.str.strip()
            # Normalize headers (remove accents, dots, extra spaces) and map common SAP Portuguese headers
            def _norm(s):
                try:
                    s0 = str(s)
                    s1 = unicodedata.normalize('NFKD', s0).encode('ascii', 'ignore').decode('ascii')
                    s1 = re.sub(r"[^0-9a-zA-Z ]+", ' ', s1)
                    return re.sub(r"\s+", ' ', s1).strip().lower()
                except Exception:
                    return str(s)

            norm_map = {c: _norm(c) for c in df_fagl.columns}

            # mapping heuristics
            # explicit pattern mappings for common SAP/Portuguese headers
            pattern_map = [
                (r'dt\.?l', 'Posting Date'),
                (r'dt\.?lcto', 'Posting Date'),
                (r'post.*date|posting', 'Posting Date'),
                # Skip auto-mapping of 'montante' - let the explicit rename_map handle dual-currency distinction
                # (r'montante', 'Amount in local currency'),
                # (r'mont\.?em', 'Amount in local currency'),
                # (r'mont\.em', 'Amount in local currency'),
                # (r'mont\.em mi', 'Amount in local currency'),
                # (r'mont\.em mi2', 'Amount in local currency'),
                # (r'mont.*mi', 'Amount in local currency'),
                (r'conta', 'Account'),
                (r'gl acct|gl acct|g/l acct', 'Account'),
                (r'empr|empresa|company code', 'Company Code'),
                (r'cen.*lucro|cen.lucro|cen lucro|profit center|centro cst', 'Profit Center'),
                # Skip 'moed' pattern - let explicit rename_map handle Moeda interna / Moeda interna 2
                # to avoid matching 'moeda' inside 'montante em moeda interna'
                # (r'moed', 'Local Currency'),
                (r'elemento pep|wbs', 'WBS element')
            ]

            rename_guess = {}
            for orig, n in norm_map.items():
                # try explicit patterns first
                for pat, target in pattern_map:
                    if re.search(pat, n):
                        rename_guess[orig] = target
                        break
                if orig in rename_guess:
                    continue
                # additional direct substring checks for SAP Portuguese headers
                if 'dt' in n and ('lcto' in n or 'lcto' in n or 'data' in n or 'dt lcto' in n or 'dtlcto' in n):
                    rename_guess[orig] = 'Posting Date'
                    continue
                if 'conta' in n or 'g l acct' in n or 'gl acct' in n or n == 'account':
                    rename_guess[orig] = 'Account'
                    continue
                if 'empr' in n or 'empresa' in n or 'company code' in n or 'company' in n:
                    # prefer Company Code if token present
                    if 'code' in n or 'codigo' in n or 'empr' in n:
                        rename_guess[orig] = 'Company Code'
                    else:
                        rename_guess[orig] = 'Company'
                    continue
                if ('cen' in n and ('lucro' in n or 'cst' in n)) or 'profit center' in n or 'centro cst' in n:
                    rename_guess[orig] = 'Profit Center'
                    continue
                # Skip auto-renaming for 'mont' columns - let the explicit rename_map handle it
                # to preserve dual-currency distinction (Montante em moeda interna vs Montante em MI2)
                # if 'mont' in n and ('mi' in n or 'amount' in n or 'montante' in n):
                #     rename_guess[orig] = 'Amount in local currency'
                #     continue
                # Match 'moed' / 'currency' BUT NOT inside 'montante em moeda interna'
                if ('moed' in n or 'currency' in n) and 'montante' not in n:
                    rename_guess[orig] = 'Local Currency'
                    continue
                if 'elemento' in n or 'wbs' in n:
                    rename_guess[orig] = 'WBS element'

            # debug: always show normalized headers sample to diagnose mapping
            try:
                st.info(f"Normalized headers (sample): {list(norm_map.items())[:12]}")
            except Exception:
                pass

            if rename_guess:
                try:
                    df_fagl = df_fagl.rename(columns=rename_guess)
                    st.info(f"Header mappings applied for Actuals: {list(rename_guess.values())}")
                    st.info(f"Rename dictionary: {rename_guess}")
                except Exception:
                    pass
            # log unmapped columns and full header list for debugging
            try:
                unmapped = [c for c in df_fagl.columns if c not in rename_guess.keys()]
                st.info(f"Unmapped Actuals columns: {unmapped}")
                st.info(f"Final Actuals columns: {list(df_fagl.columns)}")
            except Exception:
                pass
            
            # Padronização de Colunas Universal (mapear variações portuguesas atuais)
            # IMPORTANT: Keep dual-currency amounts distinct!
            # - Montante em moeda interna: Amount in local currency (CAD/USD/etc per Moeda interna)
            # - Montante em MI2: Amount in EUR (always EUR per Moeda interna 2)
            rename_map = {
                'Dt.lçto.': 'Posting Date',
                'Data de lançamento': 'Posting Date',
                'Empr': 'Company Code',
                'Empresa': 'Company Code',
                'Conta': 'Account',
                'Cont': 'Account',
                'Cen.lucro': 'Profit Center',
                'Centro de lucro': 'Profit Center',
                'Centro cst': 'Profit Center',
                'MoedI': 'Local Currency',
                'Moeda interna': 'Local Currency',
                'Moeda interna 2': 'Local Currency 2',
                'Mont.em MI': 'Montante em moeda interna',  # Keep Portuguese to preserve dual-currency logic
                'Montante em moeda interna': 'Montante em moeda interna',  # No change
                'Montante em MI2': 'Montante em MI2',  # No change - amount in EUR
                'Elemento PEP': 'WBS element'
            }
            df_fagl = df_fagl.rename(columns=rename_map)

            # handle duplicate column names which often arise when two similar
            # fields (eg. "Moeda interna" and "Moeda interna 2") are both
            # mapped to the same English header.  Instead of dropping extras we
            # rename them with a suffix so the data is preserved and the caller
            # can decide which one to use.
            if df_fagl.columns.duplicated().any():
                cols = df_fagl.columns.tolist()
                seen = {}
                for idx, col in enumerate(cols):
                    if col in seen:
                        seen[col] += 1
                        newname = f"{col}_{seen[col]}"
                        df_fagl.columns.values[idx] = newname
                        try:
                            st.warning(f"Renamed duplicate column '{col}' to '{newname}'")
                        except Exception:
                            pass
                    else:
                        seen[col] = 0
                # rebuild cols list after renaming
                df_fagl.columns = df_fagl.columns  # trigger update

            # clean numeric columns after duplicates are resolved;
            # apply to every column that starts with the amount header so we
            # don't miss renamed duplicates. Also clean Portuguese dual-currency amounts.
            for col in list(df_fagl.columns):
                if col.startswith('Amount in local currency') or col.startswith('Montante em'):
                    df_fagl[col] = df_fagl[col].apply(clean_financial_number)
            
            df_fagl = apply_fiscal_calendar(df_fagl)

            # 1.5 PLANO DE CONTAS (COA) E CROSS-MATCH
            coa_file = find_file('gl account mapping') or find_file('coa_mapping')
            if coa_file:
                df_coa = read_csv_smart(coa_file)
                if 'G/L Acct' in df_coa.columns: df_coa = df_coa.rename(columns={'G/L Acct': 'Account'})
                
                # O segredo do PROCV infalível: Limpa .0 e Zeros à esquerda das duas tabelas!
                if 'Account' in df_fagl.columns and 'Account' in df_coa.columns:
                    df_fagl['Account'] = df_fagl['Account'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.lstrip('0')
                    df_coa['Account'] = df_coa['Account'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.lstrip('0')
                    df_fagl = pd.merge(df_fagl, df_coa, on='Account', how='left')
                
            # SAFETY NET CFO: Se a conta não for mapeada, ele deduz pela classe
            # note: df_fagl['P&L LVL 5'] can return a DataFrame if there are duplicate
            # column names; .isna().all() on a DataFrame returns a Series, and using
            # that in an `if` expression causes the ambiguous truth-value error that
            # was bubbling up as "Erro ao carregar Actuals".  Make sure we reduce to a
            # single bool and log duplicates for debugging.
            if 'P&L LVL 5' not in df_fagl.columns or (
                    isinstance(df_fagl['P&L LVL 5'], pd.DataFrame)
                    and df_fagl['P&L LVL 5'].isna().all().all()
                ) or (
                    not isinstance(df_fagl['P&L LVL 5'], pd.DataFrame)
                    and df_fagl['P&L LVL 5'].isna().all()
                ):
                # if there are duplicate columns, collapse them before deducing
                if isinstance(df_fagl['P&L LVL 5'], pd.DataFrame):
                    st.warning("Duplicate 'P&L LVL 5' columns detected; collapsing to first")
                    df_fagl['P&L LVL 5'] = df_fagl['P&L LVL 5'].iloc[:, 0]
                if 'Account' in df_fagl.columns:
                    def deduce_pl(acc):
                        val = str(acc)
                        if val.startswith('7'): return 'Net Sales'
                        if val.startswith('6'): return 'Operating Expenses'
                        return 'Other'
                    df_fagl['P&L LVL 5'] = df_fagl['Account'].apply(deduce_pl)

            # NOW apply SAP sign normalization with P&L classification available
            # Handles both Montante em moeda interna and Montante em MI2 columns
            df_fagl = normalize_sap_signs(df_fagl, amount_col='Montante em moeda interna', pl_col='P&L LVL 5', account_col='Account')
            
            try:
                st.info(f"After normalize_sap_signs: {df_fagl.shape}")
                st.info(f"Normalized columns: {[c for c in df_fagl.columns if 'Normalized' in c]}")
            except Exception:
                pass

            # validar colunas importantes nos actuals
            # Check for dual-currency amount columns (at least one should exist)
            has_mi_local = 'Montante em moeda interna' in df_fagl.columns
            has_mi2_eur = 'Montante em MI2' in df_fagl.columns
            if not (has_mi_local or has_mi2_eur):
                try:
                    st.warning("Actuals data missing: 'Montante em moeda interna' and/or 'Montante em MI2' currency columns")
                except Exception:
                    pass
            validate_columns(df_fagl, ['Posting Date', 'Account', 'Profit Center'], 'Actuals')

        except Exception as e: st.error(f"Erro ao carregar Actuals: {e}")
    else:
        try:
            st.warning("Nenhum arquivo de Actuals (fagll03 / actual base) encontrado.\n" \
                       "Se você estiver rodando a app dentro da pasta `app/`, execute: `streamlit run app/app.py` a partir da raiz do repositório,\n" \
                       "ou coloque os CSVs na mesma pasta ou em uma pasta pai. Você também pode usar a aba de Upload/Simulation para injetar dados.")
        except Exception:
            pass

    # 2. FORECAST
    fc_file = find_file('01_model_revenue', excludes=['.py'])
    if fc_file:
        try:
            df_fc = read_csv_smart(fc_file)
            df_fc = df_fc.rename(columns={'Profit center (BU)': 'Profit Center', 'Company name': 'Company', 'Company code': 'Company Code'})
            df_fc = apply_fiscal_calendar(df_fc)
            validate_columns(df_fc, ['Company', 'Company Code', 'Profit Center', 'Posting Date', 'Amount in local currency'], 'Forecast')
        except Exception as e:
            st.warning(f"Erro ao ler Forecast ({fc_file}): {e}")
    else:
        try: st.info("Forecast template/data não encontrado: 01_Model_Revenue_Planning_Flat.csv")
        except Exception: pass

    # 3. CAPACITY
    cap_file = find_file('02_model_capacity', excludes=['.py'])
    if cap_file:
        try:
            df_cap = read_csv_smart(cap_file)
            df_cap = apply_fiscal_calendar(df_cap)
            validate_columns(df_cap, ['Employee ID', 'Employee Name', 'Position Title', 'Posting Date', 'Target Capacity EUR'], 'Capacity')
        except Exception as e:
            st.warning(f"Erro ao ler Capacity ({cap_file}): {e}")
    else:
        try: st.info("Capacity template/data não encontrado: 02_Model_Capacity_Flat.csv")
        except Exception: pass

    # 4. CURRENCY MAPPING
    curr_file = find_file('currency_mapping')
    if curr_file:
        try:
            df_curr = pd.read_csv(curr_file, encoding='latin1', sep=';' if ';' in open(curr_file,'r',encoding='latin1').readline() else ',')
            df_curr['Ano'], df_curr['Mes'] = df_curr['Ano'].astype(str).str.strip(), df_curr['Mes'].astype(str).str.zfill(2)
        except Exception as e:
            st.warning(f"Erro ao ler Currency Mapping ({curr_file}): {e}")
    else:
        try: st.info("Currency mapping não encontrado: Currency_Mapping.csv ou Currency_Mapping.*")
        except Exception: pass

    return df_fagl, df_fc, df_cap, df_bud, df_curr

def filter_and_convert(df, df_currency, fy_sel, mes_sel, emp_sel, pc_sel, moeda_sel, val_col='Amount in local currency', eur_base=False):
    """Apply global filters (FY, Period, Company, Profit Center) and convert amounts to target currency.
    
    For Actuals (FAGLL03), intelligently selects amount column based on target currency:
    - If moeda_sel == 'EUR': Use 'Montante em MI2' (already EUR, no conversion needed)
    - If moeda_sel != 'EUR': Use 'Montante em moeda interna' and convert via currency mapping
    
    Args:
        df: DataFrame with Fiscal Year, Fiscal Period, etc.
        df_currency: Currency mapping table with Ano, Mes, Moeda_Origem, Moeda_Destino, Taxa_Conversao
        fy_sel: Fiscal Year selected (e.g., 'FY25/26')
        mes_sel: Fiscal Period selected ('All', 'YTD', or specific month '01-Apr', etc.)
        emp_sel: Company/Entity filter
        pc_sel: Profit Center/LOB filter
        moeda_sel: Target currency ('EUR', 'CAD', 'USD', etc.)
        val_col: Default amount column name (fallback for non-Actuals data)
        eur_base: If True, base conversions from EUR; used for Capacity data
    
    Returns:
        Filtered and converted DataFrame with 'Amount Final' column
    """
    if df.empty or 'Fiscal Year' not in df.columns:
        return pd.DataFrame()
    
    mask = (df['Fiscal Year'] == fy_sel)
    if mes_sel != 'All' and mes_sel != 'YTD':
        mask &= (df['Fiscal Period'] == mes_sel)
    
    # Tolerância a cruzamentos incompletos
    if emp_sel != 'All':
        if 'Company' in df.columns:
            mask &= (df['Company'] == emp_sel)
        elif 'Company Code' in df.columns:
            mask &= (df['Company Code'] == emp_sel)
        
    if pc_sel != 'All' and 'Profit Center' in df.columns:
        # Só filtra o Profit Center se os dados existirem
        mask &= (df['Profit Center'].astype(str) == str(pc_sel))
    
    df_res = df[mask].copy()
    if df_res.empty:
        return df_res

    df_curr = df_currency[df_currency['Moeda_Destino'] == moeda_sel] if not df_currency.empty else pd.DataFrame()
    
    # === SMART AMOUNT COLUMN SELECTION FOR ACTUALS ===
    # Detect if this is Actuals data with dual-currency structure
    has_mi2_normalized = 'Montante em MI2 Normalized' in df_res.columns
    has_amount_normalized = 'Amount Normalized' in df_res.columns
    
    if has_mi2_normalized and has_amount_normalized:
        # This is Actuals data with dual-currency support AND sign normalization
        if moeda_sel == 'EUR':
            # EUR requested: use Montante em MI2 Normalized (already EUR, no conversion needed)
            df_res['Amount Final'] = df_res['Montante em MI2 Normalized'].fillna(0)
        else:
            # Non-EUR requested: use Amount Normalized (normalized version of Montante em moeda interna) and convert
            if not df_curr.empty:
                # Merge currency rates based on (Ano, Mes, Moeda_Origem where Moeda_Origem = Moeda interna)
                df_res = pd.merge(
                    df_res,
                    df_curr,
                    left_on=['Cal_Ano', 'Cal_Mes', 'Local Currency'],
                    right_on=['Ano', 'Mes', 'Moeda_Origem'],
                    how='left'
                )
                df_res['Amount Final'] = df_res['Amount Normalized'] * df_res['Taxa_Conversao'].fillna(1.0)
            else:
                # No currency mapping available, use amount as-is
                df_res['Amount Final'] = df_res['Amount Normalized'].fillna(0)
    
    # === FALLBACK FOR ACTUALS WITHOUT DUAL-CURRENCY NORMALIZED COLUMNS ===
    elif has_mi2_normalized or has_amount_normalized:
        # Has some normalization but not the full dual structure
        if 'Montante em MI2' in df_res.columns and moeda_sel == 'EUR':
            # Use EUR amount directly
            df_res['Amount Final'] = df_res['Montante em MI2 Normalized'].fillna(0) if has_mi2_normalized else df_res['Montante em MI2'].fillna(0)
        elif 'Montante em moeda interna' in df_res.columns:
            # Use local currency with conversion
            if not df_curr.empty:
                df_res = pd.merge(
                    df_res,
                    df_curr,
                    left_on=['Cal_Ano', 'Cal_Mes', 'Local Currency'],
                    right_on=['Ano', 'Mes', 'Moeda_Origem'],
                    how='left'
                )
                df_res['Amount Final'] = df_res['Amount Normalized'].fillna(0) * df_res['Taxa_Conversao'].fillna(1.0)
            else:
                df_res['Amount Final'] = df_res['Amount Normalized'].fillna(0)
    
    # === STANDARD CONVERSION FOR FORECAST/CAPACITY ===
    elif eur_base:
        # Capacity and similar data: convert from EUR base
        if moeda_sel == 'EUR' or df_curr.empty:
            df_res['Amount Final'] = df_res.get(val_col, 0).fillna(0)
        else:
            taxa_eur = df_curr[df_curr['Moeda_Origem'] == 'EUR']
            if not taxa_eur.empty:
                df_res = pd.merge(
                    df_res,
                    taxa_eur[['Ano', 'Mes', 'Taxa_Conversao']],
                    left_on=['Cal_Ano', 'Cal_Mes'],
                    right_on=['Ano', 'Mes'],
                    how='left'
                )
                df_res['Amount Final'] = df_res.get(val_col, 0).fillna(0) * df_res['Taxa_Conversao'].fillna(1.0)
            else:
                df_res['Amount Final'] = df_res.get(val_col, 0).fillna(0)
    
    else:
        # Forecast data: convert from local currency to target
        if 'Local Currency' in df_res.columns and not df_curr.empty:
            df_res = pd.merge(
                df_res,
                df_curr,
                left_on=['Cal_Ano', 'Cal_Mes', 'Local Currency'],
                right_on=['Ano', 'Mes', 'Moeda_Origem'],
                how='left'
            )
            df_res['Amount Final'] = df_res.get(val_col, 0).fillna(0) * df_res['Taxa_Conversao'].fillna(1.0)
        else:
            df_res['Amount Final'] = df_res.get(val_col, 0).fillna(0)
    
    return df_res
            
    return df_res