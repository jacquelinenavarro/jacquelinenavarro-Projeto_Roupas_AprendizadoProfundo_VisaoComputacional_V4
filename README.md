# Sistema de Reconhecimento de Categoria de Roupas para Pessoas com Deficiência Visual

## 📋 Descrição do Projeto

Este projeto implementa um sistema de classificação de imagens usando Deep Learning para auxiliar pessoas com deficiência visual a identificar categorias de roupas. O sistema utiliza Transfer Learning com 4 arquiteturas de redes neurais convolucionais (CNNs) pré-treinadas, comparando seu desempenho e otimizando o melhor modelo encontrado.

### Motivação

A ideia surgiu a partir da participação no curso "Recursos Educacionais Acessíveis para o Ensino e Aprendizagem de Ciências da Natureza e Matemática para Estudantes com Deficiência Visual" (Instituto Benjamin Constant/MEC, 2023) e foi inspirada por relatos de pessoas com deficiência visual sobre os desafios cotidianos, como escolher roupas adequadas.

## 🎯 Objetivos

- **Objetivo Principal**: Classificar imagens de roupas em 17 categorias (Blouses_Shirts, Dresses, Sweaters, Jackets_Coats, etc.)
- **Modelos Avaliados**: GoogLeNet, ResNet-50, MobileNet-v2, EfficientNet-B0
- **Abordagem**: Transfer Learning com modelos pré-treinados no ImageNet
- **Otimização**: Fine-tuning do melhor modelo com técnicas avançadas de regularização

## 📦 Resumo do Escopo Entregável

**Fase Atual do Projeto**: Esta etapa concentrou-se estritamente na validação da arquitetura de **Classificação de Categoria de Roupas (Single-Label)**, priorizando a obtenção de um baseline robusto e otimizado. A abordagem metodológica incluiu:

- ✅ Comparação sistemática de 4 arquiteturas de deep learning
- ✅ Avaliação rigorosa com múltiplas métricas (Accuracy, Precision, Recall, F1-Score)
- ✅ Otimização do melhor modelo com técnicas conservadoras de fine-tuning
- ✅ Documentação completa com visualizações e relatórios exportáveis

**Classificação Multi-Label (Roupas + Cores)**: Embora seja o objetivo final do projeto para maximizar a utilidade prática do sistema, a classificação simultânea de categoria e cor foi estrategicamente adiada para trabalhos futuros. Esta decisão metodológica permitiu:

1. Estabelecer um baseline sólido e validado para a tarefa de classificação de categorias
2. Otimizar recursos computacionais e tempo de desenvolvimento dentro das restrições do prazo acadêmico
3. Garantir a qualidade e reprodutibilidade dos resultados apresentados
4. Criar uma base técnica robusta para expansão futura do sistema

**Saídas Geradas**: Todos os resultados são automaticamente salvos em múltiplos formatos:
- 📊 **Imagens**: Gráficos de distribuição, históricos de treinamento, matrizes de confusão
- 📄 **Arquivos TXT**: Relatórios detalhados de cada etapa do processamento
- 📋 **Arquivo JSON**: Métricas baseline para comparação automática
- 📦 **Arquivo ZIP**: Pacote completo com todos os resultados para download (gerado automaticamente ao final da execução)

## 📚 Referências

**Artigo Base**: "Blind People: Clothing Category Classification and Stain Detection Using Transfer Learning" (2023)
- DOI: https://doi.org/10.3390/app13031925
- Melhor modelo no artigo: GoogLeNet

## 🗂️ Estrutura do Projeto

scripts/
├── 01_preparacao_dados.py       # Download, exploração e visualização do dataset
├── 02_preprocessamento.py       # Transformações, data augmentation e dataloaders
├── 03_modelos.py                # Definição dos quatro modelos com transfer learning
├── 04_treinamento.py            # Funções de treinamento com early stopping
├── 05_avaliacao.py              # Métricas, visualizações e relatórios
├── 06_execucao_principal.py     # Pipeline completo de treinamento baseline
├── 07_otimizacao_modelo.py      # Otimização do melhor modelo (EfficientNet-B0)
└── 08_exportacao_resultados.py  # Exportação de todos os resultados em arquivo ZIP

## 📊 Dataset

**DeepFashion-1** (Kaggle)
- Link: https://www.kaggle.com/datasets/vishalbsadanand/deepfashion-1
- Total de imagens: 11.484
- Número de categorias: 17
- Divisão: 70% treino, 15% validação, 15% teste

### Categorias de Roupas:

1. Blouses_Shirts (2.044 imagens - 17.80%)
2. Dresses (1.569 imagens - 13.66%)
3. Sweaters (1.359 imagens - 11.83%)
4. Jackets_Coats (1.149 imagens - 10.01%)
5. Tees_Tanks (1.149 imagens - 10.01%)
6. Shorts (840 imagens - 7.31%)
7. Skirts (735 imagens - 6.40%)
8. Cardigans (630 imagens - 5.49%)
9. Pants (525 imagens - 4.57%)
10. Rompers (420 imagens - 3.66%)
11. Jeans (315 imagens - 2.74%)
12. Graphic_Tees (315 imagens - 2.74%)
13. Sweatshirts (210 imagens - 1.83%)
14. Jackets_Vests (105 imagens - 0.91%)
15. Leggings (105 imagens - 0.91%)
16. Suiting (14 imagens - 0.12%)

### Como usar no Kaggle:

O dataset já está disponível no Kaggle em `/kaggle/input/deepfashion-1/`. O script detecta automaticamente o ambiente e usa o caminho correto.

## 🚀 Como Executar

### No Kaggle Notebook (Recomendado):

1. **Crie um novo notebook no Kaggle**
2. **Adicione o dataset DeepFashion-1** ao notebook
3. **Copie cada script para uma célula separada**
4. **Execute na ordem sequencial**:

\`\`\`python
# Bloco 1: Preparação de Dados
%run scripts/01_preparacao_dados.py

# Bloco 2: Pré-processamento
%run scripts/02_preprocessamento.py

# Bloco 3: Definição dos Modelos
%run scripts/03_modelos.py

# Bloco 4: Funções de Treinamento
%run scripts/04_treinamento.py

# Bloco 5: Funções de Avaliação
%run scripts/05_avaliacao.py

# Bloco 6: Treinamento Baseline (4 modelos)
%run scripts/06_execucao_principal.py

# Bloco 7: Otimização do Melhor Modelo
%run scripts/07_otimizacao_modelo.py

# Bloco 8: Exportação de Resultados em ZIP
%run scripts/08_exportacao_resultados.py
\`\`\`

### Tempo de Execução Estimado:

- **Bloco 1**: ~2 minutos (exploração do dataset)
- **Bloco 2**: ~1 minuto (pré-processamento)
- **Bloco 3**: ~30 segundos (definição dos modelos)
- **Bloco 4-5**: Instantâneo (apenas definições de funções)
- **Bloco 6**: ~2-3 horas (treinamento de 4 modelos com 10 épocas cada)
- **Bloco 7**: ~1-2 horas (otimização com 20 épocas)
- **Bloco 8**: ~30 segundos (exportação do ZIP)

**Total**: ~4-6 horas no Kaggle com GPU T4

## 📈 Métricas Avaliadas

O projeto calcula e visualiza as seguintes métricas para cada modelo:

- ✅ **Train Accuracy**: Acurácia no conjunto de treinamento
- ✅ **Validation Accuracy**: Acurácia no conjunto de validação
- ✅ **Test Accuracy**: Acurácia no conjunto de teste
- ✅ **Precision**: Precisão das predições positivas (macro-average)
- ✅ **Recall**: Taxa de verdadeiros positivos identificados (macro-average)
- ✅ **F1-Score**: Média harmônica entre Precision e Recall (macro-average)
- ✅ **Overfitting**: Diferença entre Train Accuracy e Test Accuracy
- ✅ **Matriz de Confusão**: Visualização detalhada de acertos e erros por classe
- ✅ **F1-Score por Classe**: Desempenho individual em cada categoria de roupa

## 📊 Arquivos de Saída Gerados

### Bloco 1 - Preparação de Dados:
- `bloco01_primeiras_10_imagens.png`: Amostra visual do dataset
- `bloco01_distribuicao_categorias.png`: Gráfico de distribuição das 17 categorias
- `bloco01_saida.txt`: Relatório com estatísticas do dataset

### Bloco 2 - Pré-processamento:
- `bloco02_distribuicao_categorias.png`: Distribuição após pré-processamento
- `bloco02_saida.txt`: Relatório com divisão treino/val/teste e transformações

### Bloco 6 - Treinamento Baseline:
- `bloco06_historico_googlenet.png`: Curvas de loss e acurácia (GoogLeNet)
- `bloco06_historico_resnet50.png`: Curvas de loss e acurácia (ResNet-50)
- `bloco06_historico_mobilenet_v2.png`: Curvas de loss e acurácia (MobileNet-v2)
- `bloco06_historico_efficientnet_b0.png`: Curvas de loss e acurácia (EfficientNet-B0)
- `bloco06_matriz_confusao_googlenet.png`: Matriz de confusão (GoogLeNet)
- `bloco06_matriz_confusao_resnet50.png`: Matriz de confusão (ResNet-50)
- `bloco06_matriz_confusao_mobilenet_v2.png`: Matriz de confusão (MobileNet-v2)
- `bloco06_matriz_confusao_efficientnet_b0.png`: Matriz de confusão (EfficientNet-B0)
- `bloco06_tabela_comparacao_modelos.png`: Tabela comparativa de desempenho
- `bloco06_baseline_metrics.json`: Métricas do melhor modelo (para comparação automática)
- `bloco06_saida.txt`: Relatório completo do treinamento baseline

### Bloco 7 - Otimização:
- `bloco07_historico_treinamento.png`: Curvas de loss e acurácia da otimização
- `bloco07_matriz_confusao_otimizada.png`: Matriz de confusão do modelo otimizado
- `bloco07_f1_score_por_classe.png`: Comparação de F1-Score por categoria
- `bloco07_tabela_comparacao_final.png`: Tabela comparativa (Original vs Otimizado vs TTA)
- `bloco07_saida.txt`: Relatório completo da otimização

### Bloco 8 - Exportação:
- `resultados_deepfashion_classificacao.zip`: Arquivo ZIP contendo todos os resultados acima

**Total**: 18 arquivos de imagem + 4 arquivos TXT + 1 arquivo JSON + 1 arquivo ZIP

## ⚙️ Hiperparâmetros

### Treinamento Baseline (Bloco 6):
\`\`\`python
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
OPTIMIZER = Adam
LOSS_FUNCTION = CrossEntropyLoss
TRAIN_SPLIT = 0.7   # 70% treino (8.038 imagens)
VAL_SPLIT = 0.15    # 15% validação (1.722 imagens)
TEST_SPLIT = 0.15   # 15% teste (1.724 imagens)
\`\`\`

### Otimização (Bloco 7):
\`\`\`python
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001  # Reduzido para fine-tuning
WEIGHT_DECAY = 0.0005   # Regularização L2
DROPOUT = 0.2           # Dropout no classificador
EARLY_STOPPING_PATIENCE = 7
LR_SCHEDULER = ReduceLROnPlateau (patience=3, factor=0.5)
CLASS_WEIGHTING = Inversamente proporcional à frequência
TEST_TIME_AUGMENTATION = Horizontal Flip (média de 2 predições)
\`\`\`

## 🔧 Requisitos

\`\`\`python
# Deep Learning
torch >= 2.0.0
torchvision >= 0.15.0

# Processamento de Dados
numpy >= 1.24.0
pandas >= 2.0.0
Pillow >= 9.5.0

# Visualização
matplotlib >= 3.7.0
seaborn >= 0.12.0

# Métricas e Utilidades
scikit-learn >= 1.3.0
tqdm >= 4.65.0

# Dataset
kagglehub  # Para download automático do Kaggle
\`\`\`

**Nota**: Todos os requisitos já estão disponíveis no ambiente Kaggle Notebooks.

## 📝 Notas Importantes

1. **Recursos Computacionais**: O projeto foi otimizado para rodar no Kaggle com GPU T4 (16GB VRAM)
2. **Transfer Learning**: Todos os modelos usam pesos pré-treinados do ImageNet
3. **Data Augmentation**: Aplicado apenas no conjunto de treinamento (rotação, flip, crop, ajustes de cor)
4. **Reprodutibilidade**: Seeds fixadas (`torch.manual_seed(42)`) para resultados consistentes
5. **Class Weighting**: Pesos inversamente proporcionais à frequência para lidar com desbalanceamento
6. **Early Stopping**: Implementado para prevenir overfitting (patience=5 no baseline, patience=7 na otimização)
7. **Nomenclatura de Arquivos**: Todos os arquivos têm prefixo `blocoXX_` para evitar sobrescrição

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte de uma disciplina de mestrado com os seguintes requisitos:

- ✅ Artigo de referência dos últimos 4 anos (2023)
- ✅ Comparação de 4 modelos de deep learning (GoogLeNet, ResNet-50, MobileNet-v2, EfficientNet-B0)
- ✅ Otimização do melhor modelo encontrado (EfficientNet-B0)
- ✅ Uso de transfer learning com pesos pré-treinados
- ✅ Avaliação metodológica rigorosa com múltiplas métricas
- ✅ Documentação completa com código comentado linha por linha em português
- ✅ Visualizações e relatórios exportáveis

## 🔬 Metodologia de Otimização

A otimização do melhor modelo (EfficientNet-B0) seguiu uma abordagem conservadora e comprovada:

### Técnicas Aplicadas:

1. **Class Weighting**: Pesos inversamente proporcionais à frequência das classes
2. **Learning Rate Reduzido**: 0.0001 (10x menor que o baseline)
3. **Weight Decay**: 0.0005 para regularização L2
4. **Dropout**: 0.2 no classificador final
5. **Learning Rate Scheduler**: ReduceLROnPlateau (reduz LR quando validação estagna)
6. **Early Stopping**: Patience=7 para prevenir overfitting
7. **Test-Time Augmentation (TTA)**: Horizontal flip para melhorar predições

### Estratégia de Fine-Tuning:

- **Fase Única**: Todas as camadas descongeladas desde o início
- **Learning Rate Baixo**: Preserva features pré-treinadas do ImageNet
- **Treinamento Prolongado**: 20 épocas com early stopping

## 📊 Resultados Esperados

Com base em execuções anteriores, os resultados típicos são:

### Baseline (Bloco 6):
- **GoogLeNet**: ~65-70% Test Accuracy
- **ResNet-50**: ~65-70% Test Accuracy
- **MobileNet-v2**: ~60-65% Test Accuracy
- **EfficientNet-B0**: ~68-72% Test Accuracy (melhor modelo)

### Otimização (Bloco 7):
- **Objetivo**: Melhorar F1-Score e reduzir overfitting
- **Resultado Esperado**: +1-3% Test Accuracy, +2-5% F1-Score

**Nota**: Os resultados podem variar devido à aleatoriedade no treinamento, mesmo com seeds fixadas.

## 👥 Autor

PPGIA - 2025
Disciplina: Aprendizado Profundo para Visão Computacional
Discente: Jacqueline Navarro da Silva

## 📄 Licença

Este projeto é para fins educacionais e de pesquisa acadêmica.

---

**Última Atualização**: Novembro 2025
**Versão**: 4.0 (Projeto_Roupas_Versao4 - Notebook Kaggle)
