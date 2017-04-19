from keras.layers import merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization


"""
Implementation of Inception Network v4 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.
"""

def inception_stem(input):
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2, 2))(input)
    # c = Convolution2D(32, 3, 3, activation='relu', subsample=(1, 1))(input)
    c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)

    c1 = MaxPooling2D((3, 3), strides=(2, 2))(c)
    c2 = Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2))(c)

    m = merge([c1, c2], mode='concat', concat_axis=1)

    c1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c1 = Convolution2D(96, 3, 3, activation='relu', )(c1)

    c2 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c2 = Convolution2D(64, 7, 1, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(64, 1, 7, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid')(c2)

    m2 = merge([c1, c2], mode='concat', concat_axis=1)

    p1 = MaxPooling2D((3,3), strides=(2, 2), )(m2)
    p2 = Convolution2D(96, 3, 3, activation='relu', subsample=(2,2))(m2)

    m3 = merge([p1, p2], mode='concat', concat_axis=1)
    m3 = BatchNormalization()(m3)
    m3 = Activation('relu')(m3)
    return m3


def inception_A(input):
    a1 = MaxPooling2D((3,3), strides=(1,1), border_mode='same')(input)
    a1 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(a1)

    a2 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(input)

    a3 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    a3 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a3)

    a4 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a4)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a4)

    m = merge([a1, a2, a3, a4], mode='concat', concat_axis=1)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    return m

def inception_B(input):
    b1 = MaxPooling2D((3,3), strides=(1,1), border_mode='same')(input)
    b1 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(b1)

    b2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)

    b3 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    b3 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(b3)
    b3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(b3)

    b4 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    b4 = Convolution2D(192, 1, 7, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(224, 7, 1, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(256, 7, 1, activation='relu', border_mode='same')(b4)

    m = merge([b1, b2, b3, b4], mode='concat', concat_axis=1)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    return m

def inception_C(input):
    c1 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c1 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(c1)

    c2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)

    c3 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    c3_1 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(c3)
    c3_2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(c3)

    c4 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    c4 = Convolution2D(192, 1, 3, activation='relu', border_mode='same')(c4)
    c4 = Convolution2D(224, 3, 1, activation='relu', border_mode='same')(c4)
    c4_1 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(c4)
    c4_2 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(c4)

    m = merge([c1, c2, c3_1, c3_2, c4_1, c4_2], mode='concat', concat_axis=1)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    return m

def reduction_A(input, k=192, l=224, m=96, n=96):
    r1 = MaxPooling2D((3,3), strides=(2, 2))(input)

    r2 = Convolution2D(n, 3, 3, activation='relu', subsample=(2,2))(input)

    r3 = Convolution2D(k, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(l, 3, 3, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(m, 3, 3, activation='relu', subsample=(2, 2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=1)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    return m

def reduction_B(input):
    r1 = MaxPooling2D((3, 3), strides=(2, 2))(input)

    r2 = Convolution2D(192, 1, 1, activation='relu')(input)
    r2 = Convolution2D(256, 3, 3, activation='relu', subsample=(2, 2))(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(320, 7, 1, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(256, 3, 3, activation='relu', border_mode='valid', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=1)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    return m


def create_inception_v4(input, nb_output):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_stem(input)

    # 4 x Inception A
    x = inception_A(x)
    x = inception_A(x)
    # x = inception_A(x)
    # x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    x = inception_B(x)
    x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    x = inception_C(x)
    x = inception_C(x)
    # x = inception_C(x)

    # # Average Pooling
    x = MaxPooling2D((2, 2))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    x = Dense(output_dim=nb_output, activation='softmax')(x)
    return x


if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model
    from keras.models import model_from_json
    from keras.optimizers import SGD

    nb_classes = 10575
    model_file = '/data/liubo/face/annotate_face_model/skyeye_small_inception_v4_%d_augment.model' %(nb_classes)
    weight_file = '/data/liubo/face/annotate_face_model/skyeye_small_inception_v4_%d_augment.weight' %(nb_classes)
    #
    ip = Input(shape=(3, 128, 128))

    inception_v4 = create_inception_v4(ip, nb_classes)
    model = Model(input=ip, output=inception_v4)

    open(model_file, 'w').write(model.to_json())
    model.save_weights(weight_file, overwrite=True)

    model.summary()
    # plot(model, to_file="small_inception-v4.png", show_shapes=True)

    model = model_from_json(open(model_file, 'r').read())
    # opt = RMSprop()
    # opt = Adam()
    opt = SGD(momentum=0.9, lr=0.01)
    model.compile(opt, 'categorical_crossentropy')
    print 'load weights'
    model.load_weights(weight_file)
    model.summary()
