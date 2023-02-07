# Classificador de ERPs Visuais com Redes Neurais Artificiais

Este projeto visa criar um classificador de ERPs visuais utilizando redes neurais artificiais, baseado na extração de características dos sinais de EEG. O classificador foi desenvolvido utilizando a linguagem de programação Python e a biblioteca Keras.
### Extração de Características
A extração de características foi realizada através da Discrete Wavelet Transform (DWT)  do tipo [Daubechies 4](https://wavelets.pybytes.com/wavelet/db4/) com 6 níveis. A energia dos sinais de decomposição e do último sinal de aproximação foi calculada para cada um dos 18 eletrodos.
### Ajuste de Parâmetros
O ajuste de parâmetros dos modelos Multi-Layer Perceptron (MLP) e Convolutional Neural Network (CNN) foi realizado através do [KerasTuner](https://keras.io/keras_tuner/). O objetivo do ajuste de parâmetros é obter o melhor desempenho possível dos modelos.
### Treinamento
Após o ajuste de parâmetros, os melhores modelos foram treinados utilizando validação cruzada dos dados. A acurácia dos resultados foi comparada através da estatística t de Student pareado modificada sob uma significância de 5%. 
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
## Licença
Este projeto está licenciado sob a licença [MIT](https://choosealicense.com/licenses/mit/).
