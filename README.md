📊 Análise Preditiva de Churn de Clientes - Telecom X

🔭 Visão Geral do Projeto

Este projeto foca na análise de dados de uma empresa de telecomunicações (Telecom X) para identificar os principais fatores que levam à evasão de clientes (churn). Utilizando técnicas de pré-processamento de dados e modelagem de machine learning, o objetivo é construir e avaliar modelos preditivos capazes de prever quais clientes têm maior probabilidade de cancelar seus serviços. Com base nos insights gerados pelos modelos, são propostas estratégias de retenção de clientes.

Este notebook, TelecomX_parte2(2).ipynb, representa a segunda fase do desafio, focada na preparação dos dados para modelagem, treinamento, avaliação e interpretação de modelos preditivos.

📁 Estrutura do Projeto

├── 📁 data
├── 📓 TelecomX_parte2.ipynb
└── 📖 README.md

data: possui dados limpos.csv. O conjunto de dados de entrada, já tratado e organizado na primeira parte do desafio.

TelecomX_parte2.ipynb: O Jupyter Notebook contendo todo o código Python para a análise, desde a preparação dos dados até a avaliação dos modelos e a geração de insights.


🛠️ Metodologia Aplicada

O projeto seguiu uma abordagem estruturada de análise de dados e machine learning:

   🧹 Carregamento e Preparação dos Dados:
        
        O arquivo CSV com dados previamente tratados foi carregado.
      
        Colunas de identificação, como customerID, que não agregam valor preditivo, foram eliminadas.
        
        Variáveis categóricas foram transformadas em formato numérico utilizando a técnica de one-hot encoding para compatibilidade com os algoritmos.

        Foi verificado o desbalanceamento de classes, constatando que a proporção de clientes que evadiram (churn) é de aproximadamente 26,6%.

        Para corrigir o desbalanceamento no treinamento, foi aplicada a técnica de oversampling SMOTE.

    🔍 Análise Exploratória e Seleção de Variáveis:

        Foi visualizada uma matriz de correlação para identificar as relações entre as variáveis numéricas e a variável alvo (Churn_Yes).

        Foram investigadas relações específicas, como Tipo de Contrato vs. Evasão e Gastos Totais vs. Evasão, utilizando gráficos de contagem e boxplots.

        Adicionalmente, foi realizada uma análise do Fator de Inflação de Variância (VIF) para remover variáveis com alta multicolinearidade.

    🤖 Construção dos Modelos Preditivos:

        Os dados foram padronizados (StandardScaler), uma etapa essencial para modelos sensíveis à escala como a Regressão Logística.

        O conjunto de dados foi dividido em 70% para treino e 30% para teste.

        Foram criados dois modelos distintos para prever a evasão:

            Regressão Logística: Um modelo linear, rápido e altamente interpretável, que requer padronização dos dados.

            Random Forest: Um modelo baseado em árvores, mais robusto e capaz de capturar relações não-lineares, que não é sensível à escala dos dados.

    📈 Avaliação e Análise de Resultados:

        Cada modelo foi avaliado utilizando as métricas: Acurácia, Precisão, Recall, F1-score e a Matriz de Confusão. Uma análise crítica foi feita para determinar o melhor modelo para o problema de negócio, com foco no Recall.

        Foi feita uma análise sobre a possibilidade de overfitting ou underfitting nos modelos.

        Foram analisadas as variáveis mais relevantes para cada modelo: os coeficientes para a Regressão Logística e a importância das variáveis (feature importances) para o Random Forest.

🎯 Fatores que Mais Influenciam a Evasão

A análise dos modelos revelou os seguintes fatores como os mais influentes na previsão de churn:

    1. Tipo e Duração do Contrato: Clientes com contratos mensais (Month-to-month) são o grupo de maior risco. Em contrapartida, o tempo de permanência (customer.tenure) e contratos de longo prazo (dois anos) são os mais fortes indicadores de lealdade.

    2. Padrão de Gastos: Clientes com gastos mensais (account.Charges.Monthly) elevados, especialmente aqueles com serviço de fibra ótica, apresentam uma taxa de evasão maior.

    3. Serviços Adicionais: A ausência de serviços de valor agregado, como segurança online (internet.OnlineSecurity_Yes), está associada a uma maior probabilidade de cancelamento.

    4. Método de Pagamento: O pagamento via cheque eletrônico (Electronic check) se destacou como um fator de risco.

💡 Estratégias de Retenção Propostas

Com base nos resultados, foram propostas as seguintes estratégias de negócio:

    Campanhas de Migração de Contrato:

        Ação: Criar ofertas proativas para clientes com contrato mensal, incentivando a migração para planos anuais ou de dois anos.

        Justificativa: Ataca diretamente o principal fator de risco de evasão. Um pequeno desconto na mensalidade para um contrato de longo prazo pode garantir a receita do cliente por um período maior.

    Análise de Satisfação para Clientes de Fibra Ótica:

        Ação: Realizar pesquisas de satisfação focadas nos clientes de fibra ótica com altas taxas mensais para entender suas principais queixas (preço, instabilidade, atendimento).

        Justificativa: O serviço premium da empresa (fibra) está associado à evasão. É crucial identificar e corrigir a causa raiz.

    Oferta de Pacotes de Serviços (Bundles):

        Ação: Oferecer serviços adicionais, como Segurança Online, de forma gratuita por um período experimental ou com desconto significativo para clientes em situação de risco.

        Justificativa: Clientes com mais serviços contratados têm menor probabilidade de sair. Aumentar o "ecossistema" do cliente com a empresa gera maior valor.

    Incentivo para Métodos de Pagamento Automáticos:

        Ação: Oferecer um pequeno desconto único ou um benefício para clientes que mudarem do cheque eletrônico para o pagamento com cartão de crédito automático.

        Justificativa: Reduz o atrito no pagamento e aumenta o compromisso do cliente, diminuindo um fator de risco secundário.

🚀 Como Executar o Projeto

Pré-requisitos

    Python 3.x

    Jupyter Notebook ou JupyterLab

📚 Bibliotecas Utilizadas

As principais bibliotecas utilizadas neste projeto estão listadas abaixo. Recomenda-se criar um arquivo requirements.txt com elas.

    pandas

    matplotlib

    seaborn

    numpy

    statsmodels

    scikit-learn

    imblearn
