from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import numpy as np
import time as time
import img2gif

#initialize variables
vgg_mean = np.array([103.939, 116.779, 123.68], dtype = np.float32)
content_weight = .025
style_weight = 1
cont_path = 'profilepic.jpg'
style_path = 'starrynight.jpg'
image_name = cont_path.rsplit('.')[0] + '_' + style_path.rsplit('.')[0]
image_width, image_height = load_img(cont_path).size
height = 400
width = int(image_width * height / image_height)
iterations = 51
time_start = time.time()
gif_location = 'gifpictures/*.png'
gif_save = 'examples/gifs/' + image_name + '.gif'

#FUNCTIONS 
#preprocessing to match vgg inputs
def preprocess(image):
    img = load_img(image, target_size = (height, width))
    img = img_to_array(img)

    img = np.expand_dims(img, axis = 0)
    img = vgg16.preprocess_input(img)

    return img

#deprocessing before displaying image
def deprocess(img):
    image = img.reshape((height, width, 3))
    image += vgg_mean
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype('uint8')

    return image

def content_loss(cont_activation, rand_content):
    loss = K.sum(K.square(rand_content - cont_activation))

    return loss

def gram_matrix(image):
    features = K.batch_flatten(K.permute_dimensions(image, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))

    return gram

def style_loss(style, new):
    style_gram = gram_matrix(style)
    new_gram = gram_matrix(new)
    n = 3
    m = height * width
    loss = (1 / (4. * (n ** 2) * (m ** 2))) * K.sum(K.square(new_gram - style_gram))

    return loss


#additional noise-reduction loss component
def total_variation_loss(x):
    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))

def main():

    #Get images, push through VGG model
    content_img = K.variable(preprocess(cont_path))
    style_img = K.variable(preprocess(style_path)) 
    rand_img = K.placeholder((1, height, width, 3))

    input_t = K.concatenate([content_img, style_img, rand_img], axis = 0)
    model = vgg16.VGG16(weights='imagenet', include_top = False, input_tensor = input_t)


    #CONTENT LOSS
    content_layer = model.get_layer('block4_conv2').output

    #get activations
    cont_activation = content_layer[0,:,:,:]
    rand_activation = content_layer[2,:,:,:]

    test_loss = K.variable(0.)

    #get loss between rand content and content image
    test_loss += content_weight * content_loss(cont_activation, rand_activation)



    #STYLE LOSS
    #choose layers to extract style from
    style_layers = ['block1_conv1', 'block2_conv1',
                      'block3_conv2', 'block4_conv2',
                      'block5_conv2']

    #get style loss for each layer
    for layer in style_layers:
        #test_loss += style_loss_per_layer(layer, style_layers, style_activation, rand_activation, model)
        style_layer = model.get_layer(layer).output
        style_features = style_layer[1,:,:,:]
        comb_features = style_layer[2,:,:,:]
        sl = style_loss(style_features, comb_features)
        test_loss += (style_weight / len(style_layers)) * sl

    #TOTAL VARIATION LOSS
    test_loss += total_variation_loss(rand_img)

    gradients = K.gradients(test_loss, rand_img)

    outputs = [test_loss]
    outputs += gradients

    f_outputs = K.function([rand_img], outputs)

    def losseval(x):
        x = x.reshape((1, height, width, 3))
        outputs = f_outputs([x])
        return outputs[0]

    def gradseval(x):
        x = x.reshape((1, height, width, 3))
        outputs = f_outputs([x])
        return outputs[1].flatten().astype('float64')

    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value  = losseval(x)
            grad_values = gradseval(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()
    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.


    #train on content and style images
    for i in range(iterations):
        time_iter = time.time()
        print('Iteration #%d' % (i+1))

        x, minval, infodict = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime = evaluator.grads, maxfun = 20)
        print('Current loss value:', minval)

        img = deprocess(x.copy())
        if i < 9:
            fname = 'gifpictures/00' + str(i+1) + '.png'
        else:
            fname = 'gifpictures/0' + str(i+1) + '.png'
        imsave(fname, img)

        #if last image, save in examples folder also
        if i == iterations -1:
            fname = 'examples/' + image_name + '.png'
            print(image_name)
            imsave(fname, img)

        time_end = time.time()
        print('Iteration time: ' + str(int(time_end - time_iter)) + ' seconds')
        print('Total time: ' + str(int(time_end - time_start)) + ' seconds')


    #make gif of results
    img2gif.makegif(gif_location, gif_save)
    print('Done')


if __name__ == '__main__':
    main()