from enum import Enum
from typing import List
from pydantic import BaseModel
from pydantic.fields import Field

class Types(str, Enum):
    pod = 'pod'
    app = 'app'

class Kinds(str, Enum):
    encoder = 'encoder'
    indexer = 'indexer'
    ranker = 'ranker'
    crafter = "crafter"
    segmenter = "segmenter"
    classifier = "classifier"
    evaluator = "evaluator"

class JinaManifestSchema(BaseModel):
    id : str = "https://api.jina.ai/schemas/hub/1.1.2.json"
    name: str = Field(default=...,
                      description='Human-readable title of the image')
    description: str = Field(default=...,
                             description='Human-readable description of the software packaged in the image')
    type: Types = Field(default=Types.pod,
                        title='type',
                        additionalProperties=False,
                        description='The type of the image')
    kind: Kinds = Field(default=Kinds.encoder,
                        title='kind',
                        additionalProperties=False,
                        description='The kind of the executor')
    author: str = Field(default='Jina AI Dev-Team (dev-team@jina.ai)',
                        description='Contact details of the people or organization responsible for the image (string)')
    version: int = Field(default=1,
                         description='Version of the image, it should be Semantic versioning-compatible')
    manifest_version: int = Field(default=1,
                                  description='The version of the manifest protocol')
    url : str = Field(default="https://jina.ai",
                                  description="URL to find more information on the image (string)")
    documentation : str = Field(default="https://docs.jina.ai",
                     description="URL to get documentation on the image (string)")
    vendor : str = Field(default="Jina AI Limited",
                     description="The name of the distributing entity, organization or individual (string)")
    license : str = Field(default="apache-2.0",
                     description="License under which contained software is distributed")
    platform : str = Field(default="linux/amd64",
                     description="A list of CPU architectures that your image built on")
    keywords : List = Field(default=[],
                            title='keywords',
                     description="A list of strings help user to filter and locate your package")


print(JinaManifestSchema(name='abc', description='def').schema_json(indent=2))