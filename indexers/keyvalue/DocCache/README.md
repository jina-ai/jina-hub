# DocCache Executor

This is the hub module for a Cache executor for Jina.

This checks whether a document has been indexed already. It does this checking the hash of the values of the combination of fields you want to cache on. By default, it checks the `.text` field.
