import PySimpleGUI as sg
import random
import os
import numpy as np
import re
from keras import models, layers, initializers
from PIL import Image
from skimage.color import hsv2rgb
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def generate_random_params(params):
    params['variance'] = random.randint(1, 200)
    n_layers = random.randint(1, 7)
    activations = ['relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'linear']
    for i in range(1, n_layers + 1):
        params['l{}_n'.format(i)] = str(random.randint(1, 128))
        params['l{}_a'.format(i)] = activations[random.randint(0, len(activations)-1)]
    if n_layers < 7:
        for j in range(n_layers + 1, 8):
            params['l{}_n'.format(j)] = 'None'
            params['l{}_a'.format(j)] = 'None'
    params['lout_a'] = activations[random.randint(0, len(activations)-1)]
    
    return params


def find_n_layers(params):
    n_layers = 7
    for key, val in sorted(params.items()):
        if val == 'None':
            n_layers = int(key[1]) - 1
            break
    
    return n_layers


def build_model(values, n_layers):  
    init = initializers.VarianceScaling(scale=values['variance'])
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(4,)))
    for i in range(1, n_layers + 1):
        n_neurons = int(values['l{}_n'.format(i)])
        activation = values['l{}_a'.format(i)]
        model.add(layers.Dense(n_neurons, kernel_initializer=init, activation=activation))
    model.add(layers.Dense(3, activation=values['lout_a']))
    model.compile(optimizer='rmsprop', loss='mse')
    
    return model


def create_grid(size, lat, scale = 1.0):
    x_dim, y_dim = size
    N = np.mean((x_dim, y_dim))
    x = np.linspace(- x_dim / N * scale, x_dim / N * scale, x_dim)
    y = np.linspace(- y_dim / N * scale, y_dim / N * scale, y_dim)
    X, Y = np.meshgrid(x, y)
    x = np.ravel(X).reshape(-1, 1)
    y = np.ravel(Y).reshape(-1, 1)
    r = np.sqrt(x ** 2 + y ** 2)
    Z = np.repeat(lat, x.shape[0]).reshape(-1, x.shape[0])

    return x, y, Z.T, r


def create_image(model, params, size):
    x_dim, y_dim = size
    X = np.concatenate(np.array(params), axis=1)
    pred = model.predict((X))
    print(pred.shape)
    img = []
    channels = pred.shape[1]
    for channel in range(channels):
        yp = pred[:, channel]
        yp = (yp - yp.min()) / (yp.max()-yp.min() + 0.000001)
        img.append(yp.reshape(y_dim, x_dim))
    img = np.dstack(img)
    if channels == 3: img = hsv2rgb(img)
    img = (img * 255).astype(np.uint8)

    return img



# --------------------------------- MAIN -----------------------------------------------------------

sg.theme('Dark Blue 3')  # please make your windows colorful

layout = [
    [sg.Text('Shape of image (required):')],
    [sg.Combo(['640 x 360', '854 x 480', '1280 x 720', '1920 x 1080'], key='shape')],
    [sg.Text('_'*60)],
    [sg.Button('I love random!', size=(15, 3))],
    [sg.Text('Variance:')],
    [sg.Slider(range=(1, 200), default_value=50, size=(20, 15), orientation='horizontal', key='variance')],
    [sg.Text('Customize layers')],
    [sg.Text('Layer 1:'), sg.Text('Number of neurons'), sg.InputText('8', size=(10, 15), key='l1_n'), sg.Text('Activation'), sg.Combo(['relu', 'sigmoid', 'tanh',
                                                                                                            'softmax', 'softplus', 'softsign',
                                                                                                            'selu', 'elu', 'linear'], default_value='relu', key='l1_a')],
    [sg.Text('Layer 2:'), sg.Text('Number of neurons'), sg.InputText('None', size=(10, 15), key='l2_n'), sg.Text('Activation'), sg.Combo(['relu', 'sigmoid', 'tanh',
                                                                                                            'softmax', 'softplus', 'softsign',
                                                                                                            'selu', 'elu', 'linear', 'None'], default_value='None', key='l2_a')],
    [sg.Text('Layer 3:'), sg.Text('Number of neurons'), sg.InputText('None', size=(10, 15), key='l3_n'), sg.Text('Activation'), sg.Combo(['relu', 'sigmoid', 'tanh',
                                                                                                            'softmax', 'softplus', 'softsign',
                                                                                                            'selu', 'elu', 'linear', 'None'], default_value='None', key='l3_a')],
    [sg.Text('Layer 4:'), sg.Text('Number of neurons'), sg.InputText('None', size=(10, 15), key='l4_n'), sg.Text('Activation'), sg.Combo(['relu', 'sigmoid', 'tanh',
                                                                                                            'softmax', 'softplus', 'softsign',
                                                                                                            'selu', 'elu', 'linear', 'None'], default_value='None', key='l4_a')],
    [sg.Text('Layer 5:'), sg.Text('Number of neurons'), sg.InputText('None', size=(10, 15), key='l5_n'), sg.Text('Activation'), sg.Combo(['relu', 'sigmoid', 'tanh',
                                                                                                            'softmax', 'softplus', 'softsign',
                                                                                                            'selu', 'elu', 'linear', 'None'], default_value='None', key='l5_a')],
    [sg.Text('Layer 6:'), sg.Text('Number of neurons'), sg.InputText('None', size=(10, 15), key='l6_n'), sg.Text('Activation'), sg.Combo(['relu', 'sigmoid', 'tanh',
                                                                                                            'softmax', 'softplus', 'softsign',
                                                                                                            'selu', 'elu', 'linear', 'None'], default_value='None', key='l6_a')],
    [sg.Text('Layer 7:'), sg.Text('Number of neurons'), sg.InputText('None', size=(10, 15), key='l7_n'), sg.Text('Activation'), sg.Combo(['relu', 'sigmoid', 'tanh',
                                                                                                            'softmax', 'softplus', 'softsign',
                                                                                                            'selu', 'elu', 'linear', 'None'], default_value='None', key='l7_a')],
    [sg.Text('Output layer:'), sg.Text('Activation'), sg.Combo(['relu', 'sigmoid', 'tanh',
                                                                'softmax', 'softplus', 'softsign',
                                                                'selu', 'elu', 'linear'], default_value='relu', key='lout_a')],
    [sg.Button('Generate image'), sg.Text('It can takes some time, please do not close')],
    [sg.Text('_'*60)],
    [sg.Text('Save as', key='save'), sg.Input('example.jpg', key='name'), sg.FileSaveAs(button_text='Browse')],
    [sg.Save()]
    ]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           


window = sg.Window('Art Generator', layout)

while True:  # Event Loop
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == 'I love random!':
        params = generate_random_params(values)
        params['Browse'] = 'Browse'
        for key in params.keys():
            window[key].update(params[key])

    if event == 'Generate image':
        n_layers = find_n_layers(values)
 # ------------------ Creating image ----------------------------------------       
        shape = re.findall('\d+', values['shape'])
        x = int(shape[0])
        y = int(shape[0])
        model = build_model(values, n_layers)
        lat = np.random.normal(0,1,1)
        grid = create_grid((x,y), lat, 1.0)
        image = create_image(model, grid, (x, y))
        image = image.squeeze()
        img = Image.fromarray(image,'RGB')
        cut_img = img.resize((640, 480))
        cut_img.save('example.png')
# -----------------------------------------------------------------------------
        layout_image = [[sg.Image('example.png')]]
        add_window = sg.Window('Image', layout_image)
        while True:
            add_event, add_values = add_window.read()
            if add_event == sg.WIN_CLOSED:
                break
        add_window.close()
    if event == 'Save':
        img.save(values['name'])

window.close()