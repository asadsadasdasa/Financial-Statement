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
    """Procura um arquivo CSV contendo `keyword` no nome.
    Busca na pasta atual e em até `max_up` diretórios-pai; se não encontrar, faz um walk a partir do cwd.
    Retorna o caminho absoluto do primeiro arquivo encontrado ou None.
    """
    if excludes is None: excludes = []
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
                (r'montante', 'Amount in local currency'),
                (r'mont\.?em', 'Amount in local currency'),
                (r'mont\.em', 'Amount in local currency'),
                (r'mont\.em mi', 'Amount in local currency'),
                (r'mont\.em mi2', 'Amount in local currency'),
                (r'mont\.em mi', 'Amount in local currency'),
                (r'mont.*mi', 'Amount in local currency'),
                (r'conta', 'Account'),
                (r'gl acct|gl acct|g/l acct', 'Account'),
                (r'empr|empresa|company code', 'Company Code'),
                (r'cen.*lucro|cen.lucro|cen lucro|profit center|centro cst', 'Profit Center'),
                (r'moed', 'Local Currency'),
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
                if 'mont' in n and ('mi' in n or 'amount' in n or 'montante' in n):
                    rename_guess[orig] = 'Amount in local currency'
                    continue
                if 'moed' in n or 'currency' in n:
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
                'Mont.em MI': 'Amount in local currency',
                'Montante em MI2': 'Amount in local currency',
                'Montante em moeda interna': 'Amount in local currency',
                'Elemento PEP': 'WBS element'
            }
            df_fagl = df_fagl.rename(columns=rename_map)

            if 'Amount in local currency' in df_fagl.columns:
                df_fagl['Amount in local currency'] = df_fagl['Amount in local currency'].apply(clean_financial_number)
            
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
            if 'P&L LVL 5' not in df_fagl.columns or df_fagl['P&L LVL 5'].isna().all():
                if 'Account' in df_fagl.columns:
                    def deduce_pl(acc):
                        val = str(acc)
                        if val.startswith('7'): return 'Net Sales'
                        if val.startswith('6'): return 'Operating Expenses'
                        return 'Other'
                    df_fagl['P&L LVL 5'] = df_fagl['Account'].apply(deduce_pl)

            # validar colunas importantes nos actuals
            validate_columns(df_fagl, ['Posting Date', 'Amount in local currency', 'Account', 'Profit Center'], 'Actuals')

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
    if df.empty or 'Fiscal Year' not in df.columns: return pd.DataFrame()
    
    mask = (df['Fiscal Year'] == fy_sel)
    if mes_sel != 'All' and mes_sel != 'YTD': mask &= (df['Fiscal Period'] == mes_sel)
    
    # Tolerância a cruzamentos incompletos
    if emp_sel != 'All':
        if 'Company' in df.columns: mask &= (df['Company'] == emp_sel)
        elif 'Company Code' in df.columns: mask &= (df['Company Code'] == emp_sel)
        
    if pc_sel != 'All' and 'Profit Center' in df.columns:
        # Só filtra o Profit Center se os dados existirem (No Actuals às vezes o PC é um código e no Forecast é o nome)
        mask &= (df['Profit Center'].astype(str) == str(pc_sel))
    
    df_res = df[mask].copy()
    if df_res.empty: return df_res

    df_curr = df_currency[df_currency['Moeda_Destino'] == moeda_sel] if not df_currency.empty else pd.DataFrame()
    
    if eur_base: 
        if moeda_sel == 'EUR' or df_curr.empty: df_res['Amount Final'] = df_res[val_col]
        else:
            taxa_eur = df_curr[df_curr['Moeda_Origem'] == 'EUR']
            if not taxa_eur.empty:
                df_res = pd.merge(df_res, taxa_eur[['Ano', 'Mes', 'Taxa_Conversao']], left_on=['Cal_Ano', 'Cal_Mes'], right_on=['Ano', 'Mes'], how='left')
                df_res['Amount Final'] = df_res[val_col] * df_res['Taxa_Conversao'].fillna(1.0)
            else: df_res['Amount Final'] = df_res[val_col]
    else:
        if 'Local Currency' in df_res.columns and not df_curr.empty:
            df_res = pd.merge(df_res, df_curr, left_on=['Cal_Ano', 'Cal_Mes', 'Local Currency'], right_on=['Ano', 'Mes', 'Moeda_Origem'], how='left')
            df_res['Amount Final'] = df_res[val_col] * df_res['Taxa_Conversao'].fillna(1.0)
        else: df_res['Amount Final'] = df_res.get(val_col, 0)
            
    return df_res