import pandas as pd

def gerar_tabela_moedas():
    # Cria os meses de 2025 e 2026 como exemplo
    meses = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    anos = ['2025', '2026']
    moedas_origem = ['CAD', 'USD', 'EUR']
    moeda_destino = ['EUR', 'USD'] # Para quais moedas a CFO vai querer ver o P&L
    
    dados = []
    
    for ano in anos:
        for mes in meses:
            for origem in moedas_origem:
                for destino in moeda_destino:
                    # Se a origem e destino forem iguais, a taxa é 1
                    if origem == destino:
                        taxa = 1.0
                    # Exemplo fictício de taxa (CAD para EUR e USD para EUR)
                    elif origem == 'CAD' and destino == 'EUR':
                        taxa = 0.68  # 1 CAD = 0.68 EUR
                    elif origem == 'USD' and destino == 'EUR':
                        taxa = 0.92  # 1 USD = 0.92 EUR
                    elif origem == 'EUR' and destino == 'USD':
                        taxa = 1.08  # 1 EUR = 1.08 USD
                    elif origem == 'CAD' and destino == 'USD':
                        taxa = 0.74  # 1 CAD = 0.74 USD
                    else:
                        taxa = 1.0
                        
                    dados.append({
                        'Ano': ano,
                        'Mes': mes,
                        'Moeda_Origem': origem,
                        'Moeda_Destino': destino,
                        'Taxa_Conversao': taxa
                    })

    df_moedas = pd.DataFrame(dados)
    df_moedas.to_csv('Currency_Mapping.csv', index=False)
    print("Arquivo 'Currency_Mapping.csv' gerado com sucesso!")

if __name__ == '__main__':
    gerar_tabela_moedas()