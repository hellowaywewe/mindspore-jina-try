__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import click
import os
import string
import random
import numpy as np
from jina.flow import Flow

RANDOM_SEED = 15

def config():
    os.environ['PARALLEL'] = str(1)
    os.environ['SHARDS'] = str(2)
    os.environ['TMP_DATA_DIR'] = '/tmp/jina/mnist/train'
    os.environ['COLOR_CHANNEL_AXIS'] = str(0)
    os.environ['TMP_WORKSPACE'] = os.environ.get('TMP_WORKSPACE', get_random_ws(os.environ['TMP_DATA_DIR']))
    os.environ['JINA_PORT'] = str(8080)
    os.environ['ENCODER'] = os.environ.get('ENCODER', 'jinaai/encoder.mindspore.lenet')

def get_random_ws(workspace_path, length=8):
    random.seed(RANDOM_SEED)
    letters = string.ascii_lowercase
    dn = ''.join(random.choice(letters) for i in range(length))
    return os.path.join(workspace_path, dn)


@click.command()
@click.option('--task', '-t')
@click.option('--num_docs', '-n', default=15000)
def main(task, num_docs):
    config()
    data_path = os.path.join(os.environ['TMP_DATA_DIR'], 'jpg/*.jpg')
    if task == 'index':
        f = Flow().load_config('flow-index.yml')
        with f:
            f.index_files(data_path, batch_size=64, read_mode='rb', size=num_docs)
    elif task == 'query':
        f = Flow().load_config('flow-query.yml')
        f.use_rest_gateway()
        with f:
            f.block()
    else:
        raise NotImplementedError(f'unknown task: {task}. A valid task is either `index` or `search`.')


if __name__ == '__main__':
    main()
