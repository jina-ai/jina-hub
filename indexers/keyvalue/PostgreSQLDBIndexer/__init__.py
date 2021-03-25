__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable

from jina.executors.indexers import BaseIndexer
from jina.helper import cached_property

if False:
    from ..PostgreSQLDBIndexer.postgresqldbhandler import PostgreSQLDBHandler


class PostgreSQLDBIndexer(BaseIndexer):
    """:class:`PostgreSQLDBIndexer` PostgreSQL based KV Indexer.
        Initialize the PostgreSQLDBIndexer.

        :param hostname: hostname of the machine
        :param port: the port
        :param username: the username to authenticate
        :param password: the password to authenticate
        :param database: the database name
        :param collection: the collection name
        :param args: other arguments
        :param kwargs: other keyword arguments
    """


    def __init__(self,
                 hostname: str = '127.0.0.1',
                 port: int = 5432,
                 username: str = 'default_name',
                 password: str = 'default_pwd',
                 database: str = 'default_db',
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database

    def __enter__(self):
        return self.connect()

    def connect(self):
        """Connect to the database. """

        import psycopg2
        from psycopg2 import Error, extras

        try:
            self.connection = psycopg2.connect(user=self.username,
                password=self.password,
                database=self.database_name,
                host=self.hostname,
                port=self.port)
            self.logger.info('Successfully connected to the database')
            self.create_table()
        except (Exception, Error) as error:
            self.logger.error('Error while connecting to PostgreSQL', error)
        return self

    def create_table(self):
        self.cursor = self.connection.cursor()
        try:
            self.cursor.execute("CREATE TABLE SQL (id serial PRIMARY KEY, vecs integer, metas varchar);")
            self.logger.info('Successfully table created')
        except:
            self.logger.error("Error while creating table!")



    def add(self, ids, vecs, metas, *args, **kwargs):
        raise NotImplementedError

    def update(self, ids, vecs, metas, *args, **kwargs):
        raise NotImplementedError

    def delete(self, ids, *args, **kwargs):
        raise NotImplementedError

    def dump(self, uri, shards, formats):
        raise NotImplementedError

    def __exit__(self, *args):
        """ Make sure the connection to the database is closed.
        """
        from psycopg2 import Error
        try:
            self.connection.close()
            self.cursor.close()
            print('PostgreSQL connection is closed')
        except (Exception, Error) as error:
            print('Error while closing: ', error)
