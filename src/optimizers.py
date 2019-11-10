import tensorflow as tf

def get_adam_optimizer(learning_rate=0.001,
                       beta_1=0.9,
                       beta_2=0.999,
                       epsilon=1e-7,
                       amsgrad=False,
                       name='Adam_optimizer',
                       **kwargs):
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1, beta_2=beta_2,
        epsilon=epsilon, amsgrad=amsgrad,
        name=name, **kwargs)

def get_rms_prop_optimizer(learning_rate=0.001,
                           rho=0.9,
                           momentum=.0,
                           epsilon=1e-7,
                           centered=False,
                           name='RMSprop_optimizer',
                           **kwargs):
    return tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon,
        centered=centered,
        name=name,
        **kwargs)

def get_sgd_optimizer(learning_rate=0.001,
                      momentum=.0,
                      nesterov=False,
                      name='SGD_optimizer',
                      **kwargs):
    return tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
        name=name, **kwargs)
