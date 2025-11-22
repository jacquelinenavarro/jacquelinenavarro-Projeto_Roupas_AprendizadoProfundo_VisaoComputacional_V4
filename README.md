# Visão Computacional Aplicada à Acessibilidade: Classificação de Vestuário com Redes Neurais Convolucionais para Pessoas com Deficiência Visual
---
### **Projeto Final da Disciplina:** Aprendizado Profundo para Visão Computacional
### **Discente:** Jacqueline Navarro da Silva
---
## 👕 Classificação de Roupas para Pessoas com Deficiência Visual


## 📌 Visão Geral

- 🧠 Transfer Learning com 4 CNNs: GoogLeNet, ResNet-50, MobileNet-v2, EfficientNet-B0
- 🎯 Classificação em 17 categorias (ex: Blouses_Shirts, Dresses, Jackets_Coats)
- 🔍 Dataset: DeepFashion-1 (11.484 imagens) - [Kaggle](https://www.kaggle.com/datasets/vishalbsadanand/deepfashion-1)
- 📈 Métricas: Accuracy, Precision, Recall, F1-Score, Matriz de Confusão

## ⚙️ Execução

- Ambiente: Kaggle Notebook com GPU T4
- Scripts organizados por blocos (pré-processamento, treinamento, otimização)
- Tempo estimado: 4–6 horas

## 🚀 Metodologia

- Comparação sistemática entre modelos
- Fine-tuning do melhor modelo (EfficientNet-B0)
- Técnicas aplicadas: Class Weighting, Dropout, LR Scheduler, TTA, Early Stopping

## 📦 Resultados

- Baseline para classificação Single-Label
- Multi-Label (categoria + cor) planejado para etapas futuras
- Arquivos gerados: imagens, relatórios `.txt`, métricas `.json`, pacote `.zip`

## 🎥 Protótipo Interativo

Este [vídeo](https://youtu.be/B-2n7g2g7KY) apresenta uma versão inicial do protótipo desenvolvido com o notebook `app-streamlit-v2.ipynb`. O código completo está disponível no [GitHub](https://github.com/jacquelinenavarro/jacquelinenavarro-Projeto_Roupas_AprendizadoProfundo_VisaoComputacional_V4.git).

A proposta de uma aplicação acessível com feedback de áudio em tempo real foi sugerida como trabalho futuro no artigo final. Este vídeo foi criado como complemento, apenas para fins de demonstração preliminar, não sendo exigência da disciplina.

## 📚 Referência

Artigo base: ["Blind People: Clothing Category Classification and Stain Detection"](https://doi.org/10.3390/app13031925) (2023)

## 🧪 Requisitos

```python
torch >= 2.0.0
torchvision >= 0.15.0
numpy, pandas, Pillow
matplotlib, seaborn
scikit-learn, tqdm
kagglehub
