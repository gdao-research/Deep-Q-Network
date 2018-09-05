CONFIG = {
    # Neural Network
    'filter1': 32,
    'filter2': 64,
    'filter3': 64,
    'filter4': 512,
    'size1': 8,
    'size2':4,
    'size3': 3,
    'strides1': 4,
    'strides2': 2,
    'strides3': 1,
    'dueling': False,
    'double': True,
    'grad_clip': 10,
    'optimizer': 'adam',

    # Training
    'iterations': 80000000,
    'update_target_freq': 10000,
    'train_freq': 4,
    'training_start': 50000,
    'eval_freq': 1000000,
    'eval_episodes': 50,

    # Environment
    'env_id': 'BreakoutNoFrameskip-v4',
    'discount_factor': 0.99,
    'frame_stack': 4,
    'noop_max': 30,

    # Replay Buffer
    'max_size': int(1e6),
    'batch_size': 64
}
