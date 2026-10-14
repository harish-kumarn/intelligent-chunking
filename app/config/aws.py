"""
app.config.aws
==============
AwsCredentials — static AWS credentials read from the environment / `.env`.

pydantic-settings loads `.env` into *its own* models only; it never exports the
lines into `os.environ`, so boto3 cannot see `AWS_ACCESS_KEY_ID` written there.
This model bridges the gap: when the keys are set they are handed to boto3
explicitly, when they are blank boto3 falls back to its standard chain
(env, `~/.aws` profile, IAM role) exactly as before.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["AwsCredentials"]


class AwsCredentials(BaseSettings):
    """Optional static credentials; every field blank → defer to boto3's chain."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    session_token: Optional[str] = Field(default=None, alias="AWS_SESSION_TOKEN")
    default_region: Optional[str] = Field(default=None, alias="AWS_DEFAULT_REGION")

    @property
    def is_static(self) -> bool:
        return bool(self.access_key_id and self.secret_access_key)

    def boto_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for `boto3.client`; empty when boto3 should resolve credentials itself."""
        if not self.is_static:
            return {}
        kwargs: Dict[str, Any] = {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
        }
        if self.session_token:
            kwargs["aws_session_token"] = self.session_token
        return kwargs
