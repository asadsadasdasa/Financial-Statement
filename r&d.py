import pandas as pd
import re
import os

print("🚀 Iniciando o Corretor Definitivo do Revenue Planning...")

# Impede o robô de ler os arquivos que ele mesmo criou
def find_file(keyword, excludes=None):
    if excludes is None: excludes = []
    for f in os.listdir('.'):
        if keyword.lower() in f.lower() and f.endswith('.csv'):
            if any(exc.lower() in f.lower() for exc in excludes): continue
            return f
    return None

def clean_financial_number(val):
    if pd.isna(val): return 0.0
    val_str = str(val).replace(',', '') 
    val_str = re.sub(r'[^\d\.-]', '', val_str) 
    try: return float(val_str)
    except: return 0.0

def parse_date(d_str):
    """Tradutor Universal de Datas: Aceita '2025-04-01' ou 'Feb/26'"""
    try:
        return pd.to_datetime(d_str).strftime('%Y-%m-%d')
    except:
        try:
            return pd.to_datetime(d_str, format='%b/%y').strftime('%Y-%m-%d')
        except:
            return pd.NaT

# ==========================================
# MODELO REVENUE PLANNING 
# ==========================================
rp_file = find_file('revenue planning report', excludes=['eur', 'model', 'flat', '01_', '02_', '03_'])

if not rp_file:
    print("❌ Arquivo de Revenue Planning não encontrado na pasta!")
else:
    print(f"✅ Lendo o arquivo: {rp_file}")
    try:
        with open(rp_file, 'r', encoding='latin1') as f: linhas = f.readlines()
        
        idx_scenarios = next((i for i, l in enumerate(linhas) if 'Select the Type' in l), 0)
        idx_cabecalho = next((i for i, l in enumerate(linhas) if 'Company code' in l), 1)
        
        sep = ';' if ';' in linhas[idx_cabecalho] else ','
        scenarios_raw = linhas[idx_scenarios].strip().split(sep)
        cabecalho_raw = linhas[idx_cabecalho].strip().split(sep)
        
        # Mapeamento do Cenário - Agora acha datas como 'Feb/26' e '2025'
        date_scenarios = {}
        for i, col in enumerate(cabecalho_raw):
            if re.search(r'(20\d{2}-\d{2}-\d{2}|[A-Za-z]{3}/\d{2})', col.strip()):
                scen = scenarios_raw[i] if i < len(scenarios_raw) and scenarios_raw[i].strip() != '' else 'Forecast'
                date_scenarios[col.strip()] = scen

        df_rp = pd.read_csv(rp_file, encoding='latin1', sep=sep, skiprows=idx_cabecalho, on_bad_lines='skip', low_memory=False)
        df_rp.columns = df_rp.columns.str.strip()
        
        cols_to_drop = [c for c in df_rp.columns if 'Total' in str(c) or str(c).startswith('Unnamed')]
        df_rp = df_rp.drop(columns=cols_to_drop, errors='ignore')
        
        # Identificação blindada das colunas de datas!
        date_cols = [c for c in df_rp.columns if re.search(r'(20\d{2}-\d{2}-\d{2}|[A-Za-z]{3}/\d{2})', str(c))]
        id_vars = [c for c in df_rp.columns if c not in date_cols]
        
        print(f"   📊 {len(date_cols)} colunas de meses detectadas (Ex: {date_cols[:3]})")
        
        # Dicionário de WBS
        if 'WBS element' in df_rp.columns:
            wbs_cols = [c for c in ['WBS element', 'Name', 'Company code', 'Profit center (BU)', 'Revenue Type', 'Direct/Ind./Interco', 'Countries (End Client)', 'Sales Rep'] if c in df_rp.columns]
            df_wbs = df_rp[wbs_cols].dropna(subset=['WBS element']).copy()
            df_wbs = df_wbs[df_wbs['WBS element'].astype(str).str.strip() != '']
            df_wbs = df_wbs.drop_duplicates(subset=['WBS element'], keep='first')
            df_wbs.to_csv('03_Master_WBS_Mapping.csv', index=False)
            print(f"   ✅ Master WBS Mapping: {len(df_wbs)} projetos documentados.")
        
        # Melt
        df_fc_flat = df_rp.melt(id_vars=id_vars, value_vars=date_cols, var_name='Original_Date', value_name='Amount in local currency')
        
        # Conversão de Datas e Números
        df_fc_flat['Posting Date'] = df_fc_flat['Original_Date'].apply(parse_date)
        df_fc_flat['Amount in local currency'] = df_fc_flat['Amount in local currency'].apply(clean_financial_number)
        
        # Link com Cenário
        df_fc_flat['Scenario'] = df_fc_flat['Original_Date'].map(lambda x: date_scenarios.get(str(x).strip(), 'Forecast'))
        
        # Limpeza final: remove meses onde a venda projetada foi ZERO
        df_fc_flat = df_fc_flat[df_fc_flat['Amount in local currency'] != 0].copy()
        
        # Remove a coluna temporária Original_Date
        df_fc_flat = df_fc_flat.drop(columns=['Original_Date'], errors='ignore')
        
        df_fc_flat.to_csv('01_Model_Revenue_Planning_Flat.csv', index=False)
        print(f"   ✅ Modelo Revenue Planning: {len(df_fc_flat)} linhas injetadas com SUCESSO!")

    except Exception as e:
        print(f"   ❌ Erro crítico no Revenue Planning: {e}")

print("\n🎉 Tudo pronto! Pode conferir o CSV '01_Model_Revenue_Planning_Flat.csv'. O Revenue Planning agora deve estar perfeito.")