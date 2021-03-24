from typing import Dict, Optional, Iterable

from jina.logging.logger import JinaLogger

if False:
    from psycopg2.database import Database


class PostgreSQLDBException(Exception):
    """Any errors raised by PostgreSQL.
    """


class PostgreSQLDBHandler:
    """PostgreSQL Handler to connect to the database and can apply add, update, delete and query.
    PostgreSQL has no access control by default, hence it can be used without username:password.
    """

    def __init__(self,
                 hostname: str = '127.0.0.1',
                 port: int = 5432,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 database: str = 'postgres'):
        self.logger = JinaLogger(self.__class__.__name__)
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        if self.username and self.password:
            self.connection_string = \
                f'PostgreSQL://{self.username}:{self.password}@{self.hostname}:{self.port}'
        else:
            self.connection_string = \
                f'PostgreSQL://{self.hostname}:{self.port}'

    def __enter__(self):
        return self.connect()

    def connect(self) -> 'PostgreSQLDBHandler':
        """Connect to the database.
        """
        import psycopg2

        try:
            self.connection = psycopg2.connect(database=self.database_name, host=self.hostname, port=self.port)
            self.logger.info('Successfully connected to the database')
        except psycopg2.errors.ConnectionException:
            raise PostgreSQLDBException('Database server is not available')
        except psycopg2.errors.ConfigFileError:
            raise PostgreSQLDBException('Credentials passed are not correct!')
        except psycopg2.errors.RaiseException as exp:
            raise PostgreSQLDBException(exp)
        except Exception as exp:
            raise PostgreSQLDBException(exp)
        return self

    @property
    def database(self) -> 'Database':
        """ Get database. """
        return self.connection[self.database_name]

    def query(self, key: str) -> Optional[bytes]:
        """ Queries the related document for the provided ``key``.

        :param key: id of the document
        """
        import psycopg2
        try:
            cursor = self.connection.cursor()
            # TODO find out the key value store for postgre? 
            cursor.execute()
            #cursor = self.collection.find({'_id': key})
            self.connection.commit()
            cursor_contents = cursor.fetchall()
            if cursor_contents:
                return cursor_contents
            return None
        except psycopg2.errors.RaiseException as exp:
            raise Exception(f'Got an error while finding a document in the db {exp}')

    def add(self, documents: Iterable[Dict]) -> Optional[str]:
        """ Insert the documents into the database.

        :param documents: documents to be inserted
        """
        import psycopg2
        try:
            result = self.collection.insert_many(documents)
            self.logger.debug(f'inserted {len(result.inserted_ids)} documents in the database')
            return result.inserted_ids
        except psycopg2.errors.RaiseException as exp:
            raise Exception(f'got an error while inserting a document in the db {exp}')

    def __exit__(self, *args):
        """ Make sure the connection to the database is closed.
        """
        import psycopg2
        try:
            self.connection.close()
        except psycopg2.errors.RaiseException as exp:
            raise Exception(exp)

    def delete(self, keys: Iterable[str], *args, **kwargs):
        """Delete documents from the indexer.

        :param keys: document ids to delete related documents
        """
        import psycopg2
        try:
            count = self.collection.delete_many({'_id': {'$in': list(keys)}}).deleted_count
            self.logger.debug(f'deleted {count} documents in the database')
        except psycopg2.errors.RaiseException as exp:
            raise Exception(f'got an error while deleting a document in the db {exp}')

    def update(self, keys: Iterable[str], values: Iterable[bytes]) -> None:
        """ Update the documents on the database.

        :param keys: document ids
        :param values: serialized documents
        """
        import psycopg2
        try:
            # update_many updates several keys with the same op. / data.
            # we need this instead
            count = 0
            for k, new_doc in zip(keys, values):
                new_doc = {'_id': k, 'values': new_doc}
                inserted_doc = self.collection.find_one_and_replace(
                    {'_id': k},
                    new_doc
                )
                if inserted_doc == new_doc:
                    count += 1
            self.logger.debug(f'updated {count} documents in the database')
            return
        except psycopg2.errors.RaiseException as exp:
            raise Exception(f'got an error while updating documents in the db {exp}')
