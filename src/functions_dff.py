from src.metrics import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten 
from keras.layers import BatchNormalization

def build_model_MLP(hp):
    model = Sequential()
    # Profundidade da rede
    hp_net_depth = hp.Int('net_depth', min_value=2, max_value=4, step=1)
    # Funções de ativação das camadas intermediárias
    hp_activation = hp.Choice('activation_in', values=['relu', 'tanh'])
    #Inicializadores de pesos
    hp_kernel_init = hp.Choice('kernel_init', values=['random_uniform'])

    for i in range(hp_net_depth):
        # Número de neurônios em cada camada
        hp_neurons = hp.Int('neurons_l'+str(i), min_value=32, max_value=128, step=32)
        # Camadas entrada/intermediárias
        model.add(Dense(units=hp_neurons, activation=hp_activation, kernel_initializer=hp_kernel_init))
        # model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))

    # Função de ativação da camada de saída
    hp_activation_out = hp.Choice('activation_out', values=['sigmoid', 'linear'])
    # Camada de saída
    model.add(Dense(1, activation=hp_activation_out, kernel_initializer=hp_kernel_init))
    # Otimizadores
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
    # Taxas de aprendizado
    hp.Choice('learning_rate', values=[0.9, 0.5, 0.3, 0.1, 0.01, 0.001])
    # Funções de perda
    hp_func_loss = hp.Choice('func_loss', values=['binary_crossentropy'])
    # Métricas
    hp_metrics = hp.Choice('metric', values=['accuracy'])

    model.compile(optimizer=hp_optimizer, loss=hp_func_loss, metrics=hp_metrics)
    return model

def build_model_CNN(hp, input_units):
    model = Sequential()
    # Profundidade da rede convolucional
    hp_conv_depth = hp.Int('conv_depth', min_value=1, max_value=3, step=1)
    # Funções de ativação das camadas convolucionais
    hp_act_conv = hp.Choice('act_conv', values=['relu', 'tanh'])
    # Contrução das camadas entrada/intermediárias
    conv_units = input_units
    for i in range(hp_conv_depth):

        if i == 0:
            hp_filters = hp.Int('filters_conv'+str(i), min_value=32, max_value=128, step=16)
            hp_kernel = hp.Int('k_size_conv'+str(i), min_value=2, max_value=4, step=1)
            model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel, activation=hp_act_conv,
                             input_shape=(input_units, 1)))
        else:
            if conv_units < 4:
                hp.Fixed('conv_depth', i)
                break
            else:
                hp_filters = hp.Int('filters_conv'+str(i), min_value=32, max_value=128, step=16)
                hp_kernel = hp.Int('k_size_conv'+str(i), min_value=2, max_value=4, step=1)
                model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel, activation=hp_act_conv))

        max_conv = int((conv_units - hp_kernel + 1) / 4)
        if max_conv > 1:
            hp_pool = hp.Int('pool_size_conv'+str(i), min_value=2, max_value=4, step=1)
            model.add(MaxPooling1D(pool_size=hp_pool))
            conv_units = int((conv_units - hp_kernel + 1) / hp_pool)
        else:
            conv_units = int(conv_units - hp_kernel + 1)

        model.add(BatchNormalization())

    model.add(Flatten())
    # Profundidade da rede densa
    hp_densa_depth = hp.Int('densa_depth', min_value=1, max_value=3, step=1)
    # Funções de ativação das camadas intermediárias
    hp_activation = hp.Choice('act_densa_in', values=['relu', 'tanh'])
    #Inicializadores de pesos
    hp_kernel_init = hp.Choice('kernel_init', values=['random_uniform'])

    for i in range(hp_densa_depth):
        # Número de neurônios em cada camada
        hp_neurons = hp.Int('neurons_l'+str(i), min_value=32, max_value=128, step=32)
        # Camadas densas
        model.add(Dense(units=hp_neurons, activation=hp_activation, kernel_initializer=hp_kernel_init))
    
    # Função de ativação da camada de saída
    hp_activation_out = hp.Choice('act_densa_out', values=['sigmoid', 'linear'])
    # Camada de saída
    model.add(Dense(1, activation=hp_activation_out, kernel_initializer=hp_kernel_init))
    # Taxas de aprendizado
    hp.Choice('learning_rate', values=[0.9, 0.5, 0.3, 0.1, 0.01, 0.001])
    # Otimizadores
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
    # Funções de perda
    hp_func_loss = hp.Choice('func_loss', values=['binary_crossentropy'])
    # Métricas
    hp_metrics = hp.Choice('metric', values=['accuracy'])
    
    model.compile(optimizer=hp_optimizer,loss=hp_func_loss, metrics=hp_metrics)
   
    return model
