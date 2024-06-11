# Classificação de Sinais EEG com Redes Neurais
Ferramenta para extração de características de sinais EEG usando DWT e classificação de indivíduos em grupos de controle e talentosos com base em experimentos de ERP visual.
## Introdução
Este projeto visa criar um classificador de ERPs visuais utilizando redes neurais artificiais, baseado na extração de características dos sinais de EEG. O classificador foi desenvolvido utilizando a linguagem Python e a biblioteca Keras.
## Funcionalidades
- Extração de características de EEG usando DWT
- Classificação de indivíduos em grupos de controle e talentosos
- Ajuste de parâmetros usando KerasTuner
- Validação cruzada e comparação de resultados
### Extração de Características
A extração de características foi realizada através da Discrete Wavelet Transform (DWT)  do tipo [Daubechies 4](https://wavelets.pybytes.com/wavelet/db4/) com 6 níveis. A energia dos sinais de decomposição e do último sinal de aproximação foi calculada para cada um dos 18 eletrodos.
### Ajuste de Parâmetros
O ajuste de parâmetros dos modelos Multi-Layer Perceptron (MLP) e Convolutional Neural Network (CNN) foi realizado através do [KerasTuner](https://keras.io/keras_tuner/). O objetivo do ajuste de parâmetros é obter o melhor desempenho possível dos modelos.
### Treinamento
Após o ajuste de parâmetros, os melhores modelos foram treinados utilizando validação cruzada dos dados. A acurácia dos resultados foi comparada através da estatística t de Student pareado modificada sob uma significância de 5%.
## Instalação
Para instalar as dependências necessárias, execute:
```bash
pip install -r requirements.txt
```
## Uso
Para extrair características e treinar o modelo, execute os notebooks:
- `extract_wavelet_features.ipynb`: Extrai características dos sinais de EEG.
- `classification_wavelet.ipynb`: Treina e avalia o classificador.
## Estrutura do Projeto
- `datasets/:` Contém os conjuntos de dados usados.
- `outputs_fig/:` Armazena figuras e resultados gerados.
- `results/:` Resultados dos modelos treinados.
- `src/:` Código-fonte do projeto.
## Requisitos
`keras` 2.10.0
`Keras-Preprocessing` 1.1.2
`keras-tuner` 1.1.3
`matplotlib` 3.6.0
`numpy` 1.23.4
`pandas` 1.5.1
`PyWavelets` 1.4.1
`scikeras` 0.9.0
`scikit-learn` 1.1.2
`scipy` 1.9.3
`seaborn` 0.12.2
`tensorflow` 2.10.0
## Autor
- [@abner-lucas](https://github.com/abner-lucas)
## Contribuição
Sinta-se à vontade para abrir issues e enviar pull requests. Para contribuir, siga estas etapas:
1. Faça um fork do repositório.
2. Crie um branch para sua feature (`git checkout -b feature/nome-da-feature`).
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`).
4. Faça o push para o branch (`git push origin feature/nome-da-feature`).
5. Abra um pull request.
## Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo [MIT](https://choosealicense.com/licenses/mit/) para mais detalhes.

