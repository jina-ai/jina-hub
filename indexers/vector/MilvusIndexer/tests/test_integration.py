import os
import shutil
import time

import numpy as np
import pytest
from milvus import Milvus

from .. import MilvusIndexer
from ..MilvusDBHandler import MilvusDBHandler

cur_dir = os.path.dirname(os.path.abspath(__file__))

port = 19530
host = '127.0.0.1'
img_name = 'milvusdb/milvus:0.10.0-cpu-d061620-5f3c00'
host_milvus_tmp = os.path.join(cur_dir, 'milvus_tmp')


def rm_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=None)


def create_collection(collection_name):
    client = Milvus(host, str(port))

    status, ok = client.has_collection(collection_name)
    if not ok:
        param = {
            'collection_name': collection_name,
            'dimension': 3,
        }
        client.create_collection(param)
    client.close()


def docker_run():
    import docker
    conf_tmp = os.path.join(host_milvus_tmp, 'conf')
    os.makedirs(host_milvus_tmp, exist_ok=True)
    os.makedirs(conf_tmp, exist_ok=True)
    shutil.copy(os.path.join(cur_dir, 'server_config.yaml'), conf_tmp)
    client = docker.from_env()
    client.images.pull(img_name)

    bind_volumes = {
        os.path.join(host_milvus_tmp, 'db'): {'bind': '/var/lib/milvusdb/db', 'mode': 'rw'},
        os.path.join(host_milvus_tmp, 'conf'): {'bind': '/var/lib/milvusdb/conf', 'mode': 'rw'},
        os.path.join(host_milvus_tmp, 'logs'): {'bind': '/var/lib/milvusdb/logs', 'mode': 'rw'},
        os.path.join(host_milvus_tmp, 'wal'): {'bind': '/var/lib/milvusdb/wal', 'mode': 'rw'}
    }

    container = client.containers.run(img_name, name='milvus_test_image',
                                      volumes=bind_volumes, detach=True, auto_remove=True,
                                      ports={f'{port}/tcp': f'{port}', '19121/tcp': '19121'},
                                      network_mode='host')
    client.close()
    return container


def docker_clean(container):
    container.stop()


def setup_tear_down_decorator(func):
    def wrapper():
        container = docker_run()
        func()
        docker_clean(container)
        time.sleep(2)
        rm_files([host_milvus_tmp])

    return wrapper


@pytest.mark.skip
@setup_tear_down_decorator
def test_milvusdbhandler_simple():
    collection_name = 'simple_milvus'
    create_collection(collection_name)

    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([0, 1, 2, 3]).reshape(-1, 1)
    with MilvusDBHandler(host, port, collection_name) as db:
        db.insert(keys, vectors)
        dist, idx = db.search(vectors, 2)
        dist = np.array(dist)
        idx = np.array(idx)
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(idx, np.array([[0, 1], [1, 0], [2, 1], [3, 2]]))


@pytest.mark.skip
@setup_tear_down_decorator
def test_milvusdbhandler_build():
    collection_name = 'build_milvus'
    create_collection(collection_name)

    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([0, 1, 2, 3]).reshape(-1, 1)
    with MilvusDBHandler(host, port, collection_name) as db:
        db.insert(keys, vectors)
        db.build_index(index_type='IVF,Flat', index_params={'nlist': 2})

        dist, idx = db.search(vectors, 2, {'nprobe': 2})
        dist = np.array(dist)
        idx = np.array(idx)
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(idx, np.array([[0, 1], [1, 0], [2, 1], [3, 2]]))


@pytest.mark.skip
@setup_tear_down_decorator
def test_milvus_indexer():
    collection_name = 'milvus_indexer'
    create_collection(collection_name)

    vectors = np.array([[1, 1, 1],
                        [10, 10, 10],
                        [100, 100, 100],
                        [1000, 1000, 1000]])
    keys = np.array([0, 1, 2, 3]).reshape(-1, 1)
    with MilvusIndexer(host=host, port=port,
                       collection_name=collection_name, index_type='IVF,Flat',
                       index_params={'nlist': 2}) as indexer:
        indexer.add(keys, vectors)
        idx, dist = indexer.query(vectors, 2, search_params={'nprobe': 2})
        dist = np.array(dist)
        idx = np.array(idx)
        assert idx.shape == dist.shape
        assert idx.shape == (4, 2)
        np.testing.assert_equal(idx, np.array([[0, 1], [1, 0], [2, 1], [3, 2]]))
