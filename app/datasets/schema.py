"""
app.datasets.schema
===================
The versioned CSV contracts of ground-truth datasets, one per module that has evaluation examples.

    retrieval_v1 (hybrid_retrieval) : the auto-grounding format (docs/ARTIFACTS.md `groundtruth_dataset.csv`)
                                      plus two leading identity columns, `case_id` and `document_id`, so a row
                                      can be joined to a trace (`X-Case-Id`) and to the Milvus row
                                      `{document_id}:{answer_chunk_id}`

`module_name` and `dataset_version` are not row columns: the S3 path and the manifest carry them once.
No ingestion / chunking schema exists yet — nothing in the repository produces or consumes such examples —
`SCHEMAS` is where one is registered when it does.

Columns are validated by `app.datasets.validation`; the canonical CSV written to S3 holds every column of the
schema in this order (absent optional columns empty), UTF-8, LF line endings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from app.observability.context import ID_PATTERN, ID_RULE
from app.observability.modules import ModuleName
from app.utils.naming import DOCUMENT_ID_PATTERN, DOCUMENT_ID_RULE

__all__ = ["CHUNK_ID_PATTERN", "RETRIEVAL_V1", "SCHEMAS", "Column", "DatasetSchema", "NoDatasetSchemaError", "schema_for"]

#: Merged chunk ids as `app.postprocessing.merger` writes them: `page_{n}_chunk_{m}`, `page_{a}_{b}_chunk`,
#: each optionally suffixed `_2`, `_3`, … when a page span repeats. Unscoped (no `{document_id}:` prefix).
CHUNK_ID_PATTERN = re.compile(r"^page_\d+(_\d+)?_chunk(_\d+){0,2}$")
CHUNK_ID_SEPARATOR = ";"


class NoDatasetSchemaError(ValueError):
    code = "NO_DATASET_SCHEMA"
    http_status = 422

    def __init__(self, module: str) -> None:
        super().__init__(f"module '{module}' has no dataset schema; datasets exist for: {sorted(str(m) for m in SCHEMAS)}")


@dataclass(frozen=True)
class Column:
    name: str
    required: bool = False
    kind: str = "str"                    # str | int | enum | id_list
    max_chars: int = 256
    pattern: Optional[re.Pattern] = None  # for str: the whole value; for id_list: each id
    rule: str = ""                        # human text for the pattern
    choices: Tuple[str, ...] = ()         # for enum
    description: str = ""


@dataclass(frozen=True)
class DatasetSchema:
    schema_id: str
    module: ModuleName
    purpose: str
    columns: Tuple[Column, ...]
    key_column: str = "case_id"

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def required(self) -> Tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.required)

    def column(self, name: str) -> Column:
        return next(column for column in self.columns if column.name == name)


RETRIEVAL_V1 = DatasetSchema(
    schema_id="retrieval_v1",
    module=ModuleName.HYBRID_RETRIEVAL,
    purpose="retrieval ground truth: a question and the merged chunk ids that answer it",
    columns=(
        Column("case_id", required=True, kind="str", max_chars=128, pattern=ID_PATTERN, rule=ID_RULE,
               description="unique per dataset; sent back as X-Case-Id when the row is replayed"),
        Column("document_id", required=True, kind="str", max_chars=255, pattern=DOCUMENT_ID_PATTERN, rule=DOCUMENT_ID_RULE,
               description="the published document (S3 folder / Milvus document_id) the answer lives in"),
        Column("page", kind="int", description="first source page of the answer chunks (0 when unknown)"),
        Column("type", kind="enum", choices=("direct", "indirect")),
        Column("tag", kind="enum", choices=("single", "multi_hop")),
        Column("answer_chunk_id", required=True, kind="id_list", max_chars=4096, pattern=CHUNK_ID_PATTERN,
               rule="merged chunk ids such as page_3_chunk_1 or page_3_5_chunk, ';'-separated",
               description="expected chunks (unscoped); the Milvus key is {document_id}:{answer_chunk_id}"),
        Column("question", required=True, kind="str", max_chars=4000),
        Column("answer", kind="str", max_chars=2000),
        Column("language", kind="enum", choices=("ja", "en")),
    ),
)

SCHEMAS: Dict[ModuleName, DatasetSchema] = {ModuleName.HYBRID_RETRIEVAL: RETRIEVAL_V1}


def schema_for(module: ModuleName) -> DatasetSchema:
    try:
        return SCHEMAS[ModuleName(module)]
    except (KeyError, ValueError) as exc:
        raise NoDatasetSchemaError(str(module)) from exc
