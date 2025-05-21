
# Tech Challenge - Predição de Custos Médicos

Este projeto foi desenvolvido como parte do Tech Challenge da FIAP, com o objetivo de aplicar os conhecimentos de ciência de dados para prever o **valor dos custos médicos individuais** com base em características pessoais e comportamentais.

---

## 📁 Estrutura do Projeto

```
tech-challenge-seguro/
├── data/
│   └── dados_seguros.csv       # Base de dados usada no projeto
├── images/
│   └── grafico_previsao_vs_real.png     # Gráfico de dispersão com as previsões
├── output/
│   └── relatorio_modelo.txt             # Relatório estatístico com avaliação do modelo
├── main.py                              # Script principal com toda a lógica do projeto
├── requirements.txt                     # Bibliotecas necessárias
└── README.md                            # Documentação do projeto
```

---

## 📊 Sobre os Dados

A base utilizada foi gerada sinteticamente para simular um conjunto realista de dados médicos, contendo 1000 registros com as seguintes colunas:

- `idade`: Idade do paciente (inteiro)
- `gênero`: masculino ou feminino
- `imc`: Índice de Massa Corporal (float)
- `filhos`: Quantidade de filhos (inteiro)
- `fumante`: sim ou não
- `região`: nordeste, noroeste, sudeste ou sudoeste
- `encargos`: Valor do custo médico (float)

**Observação:** As variáveis categóricas foram convertidas em numéricas usando one-hot encoding (`get_dummies`).

---

## ⚙️ Como Executar o Projeto

### 1. Clonar o repositório ou baixar os arquivos

```
git clone https://github.com/Phellype-Guilherme/Tech-challange-1-IA-fiap.git
cd tech-challenge-seguro
```


### 2. Instalar as dependências

```
pip install -r requirements.txt
```

### 3. Executar o script principal

```
python main.py
```

---

## 📈 O que será gerado

- `images/grafico_previsao_vs_real.png`: gráfico comparando previsões com os valores reais.
- `output/relatorio_modelo.txt`: relatório com métricas de avaliação e validação estatística.

---

## 🧪 Técnicas Utilizadas

- **Pré-processamento de dados:** limpeza, encoding de variáveis categóricas.
- **Modelagem:** Regressão Linear com `scikit-learn`.
- **Validação Estatística:** Análise dos coeficientes, p-values e R² com `statsmodels`.
- **Visualização:** gráfico de dispersão usando `matplotlib` e `seaborn`.

---

## 📚 Bibliotecas Usadas

- `pandas`: manipulação de dados
- `numpy`: cálculo matemático
- `matplotlib`, `seaborn`: visualização
- `scikit-learn`: machine learning (regressão, métricas)
- `statsmodels`: validação estatística

---

## 📌 Observações Finais

Este projeto demonstra como é possível utilizar técnicas simples de regressão para resolver um problema prático de previsão de custos, destacando o impacto de variáveis como tabagismo, idade e IMC nos valores cobrados por um seguro de saúde.

---

## 👥 Autores

- Nome: Phellype Guilherme Pereira da Silva - RM361625

