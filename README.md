# Financial Statement - CFO Cockpit

This Streamlit app visualizes Actuals, Forecast and Capacity data for FP&A and Delivery risk analysis.

Quick start
1. From the repository root run:

```powershell
streamlit run app/app.py
```

2. Ensure your CSV data files are present in the repository root or a parent folder. By default the app searches the current working directory and up to 3 parent directories; you may also set an explicit data directory:

```powershell
$env:DATA_DIR = "C:\path\to\data"
streamlit run app/app.py
```

Data files expected (filenames the app will try to match):
- `FAGLL03_DataBase.csv` or any file containing `fagll03` / `actual base`
- `01_Model_Revenue_Planning_Flat.csv` (or name containing `01_model_revenue`)
- `02_Model_Capacity_Flat.csv` (or name containing `02_model_capacity`)
- `COA_Mapping.csv` (or name containing `coa_mapping`)
- `Currency_Mapping.csv` (or name containing `currency_mapping`)

If a file isn't found, use the "Upload Data (Simulation)" tab to inject scenarios for testing.

If you run into encoding/separator issues, the app now tries to auto-detect delimiter and encoding (utf-8 / latin1).

Requirements

See `requirements.txt` for a minimal list. Install with:

```powershell
pip install -r requirements.txt
```

Contact
- Developer: HR Path
