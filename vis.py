import json
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model


json_file = open('model_4layer_2_2_pool.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#load weights from h5 file
model.load_weights("model_4layer_2_2_pool.h5")
layers = model.layers

print layers[0].name
print layers[0].get_weights()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
