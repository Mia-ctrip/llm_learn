from __future__ import annotations

import io
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers

from app.api import photos
from app.models.photo import Photo
from app.services.vision.quality import PhotoQualityResult, QualityModelUnavailable


def _jpeg_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (800, 1000), color=(128, 128, 128)).save(output, format="JPEG")
    return output.getvalue()


def _upload_file() -> UploadFile:
    return UploadFile(
        io.BytesIO(_jpeg_bytes()),
        filename="face.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )


def _passed_quality() -> PhotoQualityResult:
    return PhotoQualityResult(
        status="passed",
        view_type="front",
        errors=(),
        warnings=(),
        metrics={
            "face_box": {
                "left": 0.2,
                "top": 0.2,
                "right": 0.8,
                "bottom": 0.75,
                "width": 0.6,
                "height": 0.55,
                "center_x": 0.5,
                "center_y": 0.475,
            }
        },
    )


class _FakeQuery:
    def filter(self, *args: Any) -> "_FakeQuery":
        return self

    def first(self) -> None:
        return None


class _FakeDB:
    def __init__(self) -> None:
        self.added: list[Any] = []
        self.commit_count = 0
        self.rollback_count = 0

    def query(self, _model: Any) -> _FakeQuery:
        return _FakeQuery()

    def add(self, row: Any) -> None:
        self.added.append(row)

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def refresh(self, row: Any) -> None:
        if row.id is None:
            row.id = 7


class _FakeStorage:
    def __init__(self) -> None:
        self.puts: list[tuple[str, bytes, str]] = []
        self.deleted: list[str] = []

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.puts.append((key, data, content_type))

    def delete(self, key: str) -> None:
        self.deleted.append(key)

    def signed_url(self, key: str) -> SimpleNamespace:
        return SimpleNamespace(
            url=f"http://test/{key}",
            expires_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
        )


@pytest.mark.asyncio
async def test_check_in_upload_rejects_failed_quality(monkeypatch) -> None:
    failed = PhotoQualityResult(
        status="failed",
        view_type="front",
        errors=("face_cut_off",),
        warnings=(),
        metrics={"face_count": 1},
    )
    monkeypatch.setattr(photos, "_validate_check_in_target", lambda *args, **kwargs: None)
    monkeypatch.setattr(photos, "assess_photo_quality", lambda *args, **kwargs: failed)

    with pytest.raises(HTTPException) as exc_info:
        await photos.upload_photo(
            file=_upload_file(),
            taken_at=None,
            check_in_id=1,
            view_type="front",
            client_request_id=None,
            current_user=SimpleNamespace(id=1),
            db=object(),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["errors"] == ["face_cut_off"]
    assert "额头、两颊和下巴" in exc_info.value.detail["quality_meta"]["error_messages"][0]


@pytest.mark.asyncio
async def test_check_in_upload_reports_missing_quality_model(monkeypatch) -> None:
    monkeypatch.setattr(photos, "_validate_check_in_target", lambda *args, **kwargs: None)

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise QualityModelUnavailable("model missing")

    monkeypatch.setattr(photos, "assess_photo_quality", unavailable)

    with pytest.raises(HTTPException) as exc_info:
        await photos.upload_photo(
            file=_upload_file(),
            taken_at=None,
            check_in_id=1,
            view_type="front",
            client_request_id=None,
            current_user=SimpleNamespace(id=1),
            db=object(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["message"] == "photo quality model unavailable"


@pytest.mark.asyncio
async def test_passed_check_in_upload_stores_original_and_normalized(monkeypatch) -> None:
    db = _FakeDB()
    storage = _FakeStorage()
    monkeypatch.setattr(photos, "_validate_check_in_target", lambda *args, **kwargs: None)
    monkeypatch.setattr(photos, "assess_photo_quality", lambda *args, **kwargs: _passed_quality())
    monkeypatch.setattr(photos, "get_storage", lambda: storage)
    monkeypatch.setattr(
        photos,
        "_build_storage_key",
        lambda user_id, ext, now: "photos/1/test.jpg",
    )

    response = await photos.upload_photo(
        file=_upload_file(),
        taken_at=None,
        check_in_id=1,
        view_type="front",
        client_request_id=None,
        current_user=SimpleNamespace(id=1),
        db=db,
    )

    photo = next(row for row in db.added if isinstance(row, Photo))
    assert response.quality_status == "passed"
    assert photo.processed_storage_key == "photos/1/test.normalized.jpg"
    assert photo.quality_meta["normalization"]["geometry_only"] is True
    assert [item[0] for item in storage.puts] == [
        "photos/1/test.jpg",
        "photos/1/test.normalized.jpg",
    ]
    assert storage.puts[1][2] == "image/jpeg"
    assert db.commit_count == 1
