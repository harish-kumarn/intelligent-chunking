"""
app.storage.s3_client
=====================
`build_s3_client` — the one way the service builds a boto3 S3 client.

    credentials : `AwsCredentials` (env / `.env`) when static keys are set, else boto3's standard chain
    region      : the caller's, else `AWS_DEFAULT_REGION`
    config      : `S3ClientOptions` — connect / read timeouts and botocore's "standard" retry mode, so a
                  network fault surfaces as an error instead of a hung ingestion or request

Both stores (`S3ArtifactStore`, `S3DatasetStore`) call this lazily on first use; tests monkeypatch it
to hand back a fake client.
"""

from __future__ import annotations

from typing import Any, Optional

import boto3
from botocore.config import Config as BotoConfig

from app.config.aws import AwsCredentials, S3ClientOptions

__all__ = ["build_s3_client"]


def build_s3_client(region: Optional[str] = None, credentials: Optional[AwsCredentials] = None, options: Optional[S3ClientOptions] = None) -> Any:
    credentials = credentials or AwsCredentials()
    options = options or S3ClientOptions()
    config = BotoConfig(
        connect_timeout=options.connect_timeout_seconds,
        read_timeout=options.read_timeout_seconds,
        retries={"mode": "standard", "max_attempts": options.max_attempts},
    )
    return boto3.client("s3", region_name=region or credentials.default_region, config=config, **credentials.boto_kwargs())
