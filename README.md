# Classificador de ERPs Visuais com Redes Neurais Artificiais

Este projeto visa criar um classificador de ERPs visuais utilizando redes neurais artificiais, baseado na extração de características dos sinais de EEG.

### Extração de Características
A extração de características foi realizada através da Discrete Wavelet Transform (DWT)  do tipo [Daubechies 4](https://wavelets.pybytes.com/wavelet/db4/) com 6 níveis. A energia dos sinais de decomposição e do último sinal de aproximação foi calculada para cada um dos 18 eletrodos.

### Ajuste de Parâmetros
O ajuste de parâmetros dos modelos Multi-Layer Perceptron (MLP) e Convolutional Neural Network (CNN) foi realizado através do [KerasTuner](https://keras.io/keras_tuner/). O objetivo do ajuste de parâmetros é obter o melhor desempenho possível dos modelos.

### Treinamento e Validação Cruzada
Após o ajuste de parâmetros, os melhores modelos foram treinados utilizando validação cruzada dos dados. A acurácia dos resultados foi comparada através da estatística t de Student pareado modificada sob uma significância de 5%.
## Requisitos
- Python 3.x
- TensorFlow 2.x
- Numpy
- Matplotlib
## Autor

- [@abner-lucas](https://github.com/abner-lucas)


## Licença

Este projeto está licenciado sob a licença [MIT](https://choosealicense.com/licenses/mit/). Veja o arquivo LICENSE.md para mais detalhes.
