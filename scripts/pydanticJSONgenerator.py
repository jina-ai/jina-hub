import enum
from enum import Enum
from typing import List
from pydantic import BaseModel
from pydantic.fields import Field

Types = enum.Enum(
    'Types',
    {
        'pod': 'pod',
        'app': 'app'
    }
)

Kinds = enum.Enum(
    'Kinds',
    {
        'encoder' : 'encoder',
        'indexer' : 'indexer',
        'ranker' : 'ranker',
        'crafter' : 'crafter',
        'segmenter' : 'segmenter',
        'classifier' : 'classifier',
        'evaluator' : 'evaluator'
    }
)

Platform = enum.Enum(
    'Platform',
    {
        'linux/amd64': 'linux/amd64',
        'linux/arm64': 'linux/arm64',
        'linux/ppc64le': 'linux/ppc64le',
        'linux/s390x': 'linux/s390x',
        'linux/386': 'linux/386',
        'linux/arm/v7': 'linux/arm/v7',
        'linux/arm/v6': 'linux/arm/v6',
    }
)

License = enum.Enum(
    'License',
    {
        'apache-2.0': ' Apache license 2.0',
        'afl-3.0': ' Academic Free License v3.0',
        'artistic-2.0': ' Artistic license 2.0',
        'bsl-1.0': ' Boost Software License 1.0',
        'bsd-2-clause': ' BSD 2-clause "Simplified" license',
        'bsd-3-clause': ' BSD 3-clause "New" or "Revised" license',
        'bsd-3-clause-clear': ' BSD 3-clause Clear license',
        'cc': ' Creative Commons license family',
        'cc0-1.0': ' Creative Commons Zero v1.0 Universal',
        'cc-by-4.0': ' Creative Commons Attribution 4.0',
        'cc-by-sa-4.0': ' Creative Commons Attribution Share Alike 4.0',
        'wtfpl': ' Do What The F*ck You Want To Public License',
        'ecl-2.0': ' Educational Community License v2.0',
        'epl-1.0': ' Eclipse Public License 1.0',
        'eupl-1.1': ' European Union Public License 1.1',
        'agpl-3.0': ' GNU Affero General Public License v3.0',
        'gpl': ' GNU General Public License family',
        'gpl-2.0': ' GNU General Public License v2.0',
        'gpl-3.0': ' GNU General Public License v3.0',
        'lgpl': ' GNU Lesser General Public License family',
        'lgpl-2.1': ' GNU Lesser General Public License v2.1',
        'lgpl-3.0': ' GNU Lesser General Public License v3.0',
        'isc': ' ISC',
        'lppl-1.3c': ' LaTeX Project Public License v1.3c',
        'ms-pl': ' Microsoft Public License',
        'mit': ' MIT',
        'mpl-2.0': ' Mozilla Public License 2.0',
        'osl-3.0': ' Open Software License 3.0',
        'postgresql': ' PostgreSQL License',
        'ofl-1.1': ' SIL Open Font License 1.1',
        'ncsa': ' University of Illinois/NCSA Open Source License',
        'unlicense': ' The Unlicense',
        'zlib': ' zLib License',
    },
)


class JinaManifestSchema(BaseModel):
    id: str = "https://api.jina.ai/schemas/hub/1.1.2.json"
    name: str = Field(default=..., description='Human-readable title of the image')
    description: str = Field(
        default=...,
        description='Human-readable description of the software packaged in the image',
    )
    type: Types = Field(
        default=Types.pod,
        title='type',
        additionalProperties=False,
        description='The type of the image',
    )

    kind: Kinds = Field(
        default=Kinds.encoder,
        title='kind',
        additionalProperties=False,
        description="The kind of the executor",
    )

    author: str = Field(
        default='Jina AI Dev-Team (dev-team@jina.ai)',
        description='Contact details of the people or organization responsible for the image (string)',
    )
    version: int = Field(
        default=1,
        description='Version of the image, it should be Semantic versioning-compatible',
    )
    manifest_version: int = Field(
        default=1, description='The version of the manifest protocol'
    )
    url: str = Field(
        default="https://jina.ai",
        description="URL to find more information on the image (string)",
    )
    documentation: str = Field(
        default="https://docs.jina.ai",
        description="URL to get documentation on the image (string)",
    )
    vendor: str = Field(
        default="Jina AI Limited",
        description="The name of the distributing entity, organization or individual (string)",
    )
    license: License = Field(
        default="apache-2.0",
        description="License under which contained software is distributed",
    )
    platform: Platform = Field(
        default="linux/amd64",
        description="A list of CPU architectures that your image built on",
    )
    keywords: List[str] = Field(
        default=[],
        title='keywords',
        description="A list of strings help user to filter and locate your package",
    )


if __name__ == "__main__":
    print(JinaManifestSchema(name='abc', description='def').schema_json(indent=2))
