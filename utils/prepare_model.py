# Zhu Zhi, @Fairy Devices Inc., 2020
# ==============================================================================
import tensorflow as tf


class GradReverse(tf.keras.layers.Layer):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def get_config(self):
        return {}

    def call(self, x):
        return self.grad_reverse(x)

    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)

        def custom_grad(dy):
            return -self.weight*dy
        return y, custom_grad


def acrnn(input_shape,
          mel_norm_layer=None,
          init_mode='custom',
          norm='batch',
          conv1k=128,
          conv1f1=3,
          conv1f2=3,
          maxp1f1=2,
          maxp1f2=4,
          nConv2=1,
          conv2k=128,
          conv2f1=3,
          conv2f2=3,
          maxp2f1=1,
          maxp2f2=2,
          lin=768,
          rnn=128,
          rnn_drop=0.3,
          att_layer='keras0',
          att1=512,
          att2=8,
          num_heads=8,
          key_dim=128,
          **kwargs):
    """Attention-based Convolutional Recurrent Neural Network \
    Args: \
      input_shape: input tensor shape \
      mel_norm_layer: layer to normalize melspectrogram input \
      init_mode: mode of initializers \
      norm: method of normalization method \
        `batch`: batch normalization \
        `layer`: layer normalization \
      conv1: (kernel size, (filter size)) of the first cnn layer \
      maxp1: size of the first max pooling layer \
      nConv2: number of the following cnn layers \
      conv2: (kernel size, (filter size)) of the following cnn layer \
      maxp2: size of the second max pooling layer \
      lin: kernel size of the Time Distributed Linear layer \
        skip linear layer if `false` \
      rnn: kernel size of bi-LSTM layer \
      rnn_drop: dropout of bi-LSTM layer \
      att_layer: mode of attention layers \
        `custom`: custom attention layer \
        `keras`: tf.keras.layers.MultiHeadAttention() \
      att: kernel size of attention layer \
    Returns:
      tf.keras.model of ACRNN model
    """
    # input_shape: (time, n_mels, feature)
    # weight and bias init methods
    if init_mode == 'custom':
        kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        bias_init = tf.keras.initializers.Constant(0.1)
        init = {"kernel_initializer": kernel_init,
                "bias_initializer": bias_init}
    else:
        init = {}
    # normalization method
    if norm == 'batch':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm == 'layer':
        norm_layer = tf.keras.layers.LayerNormalization
    else:
        norm_layer = None
        print('Wrong normalization method')
        exit(1)
    # input layer
    x = tf.keras.layers.Input(shape=input_shape, name='input_layer')
    if mel_norm_layer:
        x_norm = mel_norm_layer(x)
    else:
        x_norm = x
    # CNN layer1
    conv = tf.keras.layers.Conv2D(
        conv1k, (conv1f1, conv1f2), padding="same",
        name='Conv0', **init)(x_norm)
    conv = norm_layer()(conv)
    conv = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)
    conv = tf.keras.layers.MaxPooling2D((maxp1f1, maxp1f2))(conv)
    # CNN layers
    for n in range(nConv2):
        conv = tf.keras.layers.Conv2D(
            conv2k, (conv2f1, conv2f2), padding="same",
            name='Conv{}'.format(n+1), **init)(conv)
        conv = norm_layer()(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)
    conv = tf.keras.layers.MaxPooling2D((maxp2f1, maxp2f2))(conv)
    # linear layer
    d1, d2, d3 = conv.shape[1:]
    linear = tf.keras.layers.Reshape((int(d1), int(d2*d3)))(conv)
    if lin:
        linear = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(lin, name='TDLin', **init))(linear)
        linear = norm_layer()(linear)
    brnn = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            rnn, dropout=rnn_drop,
            return_sequences=True, name='BLSTM'))(linear)
    # Bidirectional-LSTM and attention layer
    if att_layer == 'custom':
        # Attention layer
        u = tf.keras.layers.Dense(
            att1, activation="tanh", use_bias=False,
            name='att0')(brnn)
        u = tf.keras.layers.Dense(
            att2, use_bias=False, name='att1')(u)
        alphas = tf.keras.layers.Activation(
            'softmax', name='attention_weights')(u)
        context = tf.keras.layers.Dot(axes=1)([alphas, brnn])
        context = tf.keras.layers.Flatten()(context)
    elif att_layer.startswith('keras'):
        context = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=rnn*2, dropout=rnn_drop,
            name='mhatt')(brnn, brnn)
        if att_layer == 'keras0':
            context = tf.keras.layers.Flatten()(context)
        elif att_layer == 'keras1':
            context = tf.keras.layers.Concatenate(-1)([context, brnn])
            context = tf.keras.layers.Flatten()(context)
        elif att_layer == 'keras2':
            context = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(rnn))(context)
        elif att_layer == 'keras3':
            context = tf.keras.layers.Concatenate(-1)([context, brnn])
            context = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(rnn))(context)
        elif att_layer == 'keras4':
            context = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(rnn, return_sequences=True))(context)
            context = tf.keras.layers.Flatten()(context)
        elif att_layer == 'keras5':
            context = tf.keras.layers.Concatenate(-1)([context, brnn])
            context = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(rnn, return_sequences=True))(context)
            context = tf.keras.layers.Flatten()(context)
    else:
        context = tf.keras.layers.Flatten()(brnn)
    # output
    model = tf.keras.models.Model(inputs=x, outputs=context, name='ACRNN')
    return model


def mlp_classifier(input_shape,
                   nclass,
                   classifier_name,
                   init_mode='custom',
                   norm='batch',
                   nfc_layers=2,
                   nfc=128,
                   fc_drop=0.3,
                   **kwargs):
    '''MLP classifier \
    Arg: \
        input_shape: tuple of the shape of feature \
        nclass: number of classes \
        classifier_name: name of this classifier \
        init_mode: mode of initializers \
        norm: method of normalization method \
            `batch`: batch normalization \
            `layer`: layer normalization \
        nfc_layers: number of layers \
        nfc: number of fc features \
        fc_drop: dropout of fc layers \
    Reutrn: \
        MLP model
    '''
    if init_mode == 'custom':
        kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        bias_init = tf.keras.initializers.Constant(0.1)
        init = {"kernel_initializer": kernel_init,
                "bias_initializer": bias_init}
    else:
        init = {}
    # normalization method
    if norm == 'batch':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm == 'layer':
        norm_layer = tf.keras.layers.LayerNormalization
    else:
        norm_layer = None
        print('Wrong normalization method')
        exit(1)
    x = tf.keras.layers.Input(shape=input_shape)
    # full connected layers
    for n in range(nfc_layers):
        context = tf.keras.layers.Dense(
            nfc, name='fc{}'.format(n), **init)(x)
        context = norm_layer()(context)
        context = tf.keras.layers.LeakyReLU(alpha=0.01)(context)
        context = tf.keras.layers.Dropout(fc_drop)(context)
    # output layer
    output = tf.keras.layers.Dense(
            nclass, activation='softmax', name='softmax')(context)
    model = tf.keras.models.Model(
        inputs=x, outputs=output, name=classifier_name)
    return model


def emotion_mono(input_shape,
                 nclass,
                 mel_norm_layer=None,
                 **kwargs):
    '''Emotion classifier model with mono corpus'''
    acrnn_encoder = acrnn(input_shape, mel_norm_layer=mel_norm_layer, **kwargs)
    x = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    f = acrnn_encoder(x)
    emotion_classifier = mlp_classifier(
        f.shape[1:], nclass, classifier_name='emotion_classifier', **kwargs)
    y_p = emotion_classifier(f)
    model = tf.keras.models.Model(inputs=x, outputs=y_p)
    return model


def emotion_joint(input_shape,
                  nclasses,
                  mel_norm_layer=None,
                  **kwargs):
    '''Joint training multi-corpus model \
    Args: \
        input_shape: input shape \
        nclasses: list of number of classes for each corpus \
        corpora: list of names of corpora \
    '''
    acrnn_encoder = acrnn(input_shape, mel_norm_layer=mel_norm_layer, **kwargs)
    x = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    f = acrnn_encoder(x)
    emotion_classifier, y_ps = [], []
    for c in range(len(nclasses)):
        emotion_classifier.append(mlp_classifier(
            f.shape[1:], nclasses[c],
            classifier_name='emo{}'.format(c), **kwargs))
        y_ps.append(emotion_classifier[c](f))
    model = tf.keras.models.Model(inputs=x, outputs=y_ps)
    return model


def emotion_multi(input_shape,
                  nclasses,
                  mel_norm_layer=None,
                  **kwargs):
    '''multi-task multi-corpus model \
    Args: \
        input_shape: input shape \
        nclasses: list of number of classes for each corpus \
        corpora: list of names of corpora \
    '''
    acrnn_encoder = acrnn(input_shape, mel_norm_layer=mel_norm_layer, **kwargs)
    x = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    f = acrnn_encoder(x)
    emotion_classifier, y_ps = [], []
    for c in range(len(nclasses)):
        emotion_classifier.append(mlp_classifier(
            f.shape[1:], nclasses[c],
            classifier_name='emo{}'.format(c), **kwargs))
        y_ps.append(emotion_classifier[c](f))
    corpus_classifier = mlp_classifier(
        f.shape[1:], len(nclasses), classifier_name='corpus', **kwargs)
    y_ps.append(corpus_classifier(f))
    model = tf.keras.models.Model(inputs=x, outputs=y_ps)
    return model


def emotion_adv(input_shape,
                nclasses,
                mel_norm_layer=None,
                grl_w=1e-5,
                **kwargs):
    '''adversarial multi-corpus model \
    Args: \
        input_shape: input shape \
        nclasses: list of number of classes for each corpus \
        corpora: list of names of corpora \
        grl_w: weight of GRL layer
    '''
    acrnn_encoder = acrnn(input_shape, mel_norm_layer=mel_norm_layer, **kwargs)
    x = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    f = acrnn_encoder(x)
    emotion_classifier, y_ps = [], []
    for c in range(len(nclasses)):
        emotion_classifier.append(mlp_classifier(
            f.shape[1:], nclasses[c],
            classifier_name='emo{}'.format(nclasses[c]), **kwargs))
        y_ps.append(emotion_classifier[c](f))
    corpus_classifier = mlp_classifier(
        f.shape[1:], len(nclasses), classifier_name='corpus', **kwargs)
    grl = GradReverse(grl_w)
    f_dann = grl(f)
    y_ps.append(corpus_classifier(f_dann))
    model = tf.keras.models.Model(inputs=x, outputs=y_ps)
    return model
