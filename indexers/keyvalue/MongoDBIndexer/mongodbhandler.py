from typing import Dict, Optional, Iterable

from jina.logging.logger import JinaLogger

if False:
    from pymongo.database import Database
    from pymongo.collection import Collection


class MongoDBException(Exception):
    """Any errors raised by MongoDb.
    """


class MongoDBHandler:
    """Mongodb Handler to connect to the database and can apply add, update, delete and query.
    MongoDB has no access control by default, hence it can be used without username:password.
    """

    def __init__(self,
                 hostname: str = '127.0.0.1',
                 port: int = 27017,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 database: str = 'defaultdb',
                 collection: str = 'defaultcol'):
        self.logger = JinaLogger(self.__class__.__name__)
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.collection_name = collection
        if self.username and self.password:
            self.connection_string = \
                f'mongodb://{self.username}:{self.password}@{self.hostname}:{self.port}'
        else:
            self.connection_string = \
                f'mongodb://{self.hostname}:{self.port}'

    def __enter__(self):
        return self.connect()

    def connect(self) -> 'MongoDBHandler':
        """Connect to the database.
        """
        import pymongo
        try:
            self.client = pymongo.MongoClient(self.connection_string)
            self.client.admin.command('ismaster')
            self.logger.info('Successfully connected to the database')
        except pymongo.errors.ConnectionFailure:
            raise MongoDBException('Database server is not available')
        except pymongo.errors.ConfigurationError:
            raise MongoDBException('Credentials passed are not correct!')
        except pymongo.errors.PyMongoError as exp:
            raise MongoDBException(exp)
        except Exception as exp:
            raise MongoDBException(exp)
        return self

    @property
    def database(self) -> 'Database':
        """ Get database. """
        return self.client[self.database_name]

    @property
    def collection(self) -> 'Collection':
        """ Get collection. """
        return self.database[self.collection_name]

    def query(self, key: str) -> Optional[bytes]:
        """ Queries the related document for the provided ``key``.

        :param key: id of the document
        """
        import pymongo
        try:
            cursor = self.collection.find({'_id': key})
            cursor_contents = list(cursor)
            if cursor_contents:
                return cursor_contents[0]
            return None
        except pymongo.errors.PyMongoError as exp:
            raise Exception(f'Got an error while finding a document in the db {exp}')

    def add(self, documents: Iterable[Dict]) -> Optional[str]:
        """ Insert the documents into the database.

        :param documents: documents to be inserted
        """
        import pymongo
        try:
            result = self.collection.insert_many(documents)
            self.logger.debug(f'inserted {len(result.inserted_ids)} documents in the database')
            return result.inserted_ids
        except pymongo.errors.PyMongoError as exp:
            raise Exception(f'got an error while inserting a document in the db {exp}')

    def __exit__(self, *args):
        """ Make sure the connection to the database is closed.
        """
        import pymongo
        try:
            self.client.close()
        except pymongo.errors.PyMongoError as exp:
            raise MongoDBException(exp)

    def delete(self, keys: Iterable[str], *args, **kwargs):
        """Delete documents from the indexer.

        :param keys: document ids to delete related documents
        """
        import pymongo
        try:
            count = self.collection.delete_many({'_id': {'$in': list(keys)}}).deleted_count
            self.logger.debug(f'deleted {count} documents in the database')
        except pymongo.errors.PyMongoError as exp:
            raise Exception(f'got an error while deleting a document in the db {exp}')

    def update(self, keys: Iterable[str], values: Iterable[bytes]) -> None:
        """ Update the documents on the database.

        :param keys: document ids
        :param values: serialized documents
        """
        import pymongo
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
        except pymongo.errors.PyMongoError as exp:
            raise Exception(f'got an error while updating documents in the db {exp}')
