from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.patch_lineage import PatchLineage, PatchLineageSnapshot
from app.services.vision import tracker


class _ScalarResult:
    def __init__(self, rows: list[Any]):
        self._rows = rows

    def scalars(self) -> "_ScalarResult":
        return self

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, execute_results: list[list[Any]]):
        self._execute_results = list(execute_results)
        self.added: list[Any] = []
        self.commit_count = 0
        self._next_lineage_id = 100
        self._next_snapshot_id = 1000

    def execute(self, _stmt: Any) -> _ScalarResult:
        assert self._execute_results, "unexpected database query"
        return _ScalarResult(self._execute_results.pop(0))

    def add(self, row: Any) -> None:
        if isinstance(row, PatchLineage) and row.id is None:
            row.id = self._next_lineage_id
            self._next_lineage_id += 1
        if isinstance(row, PatchLineageSnapshot) and row.id is None:
            row.id = self._next_snapshot_id
            self._next_snapshot_id += 1
        self.added.append(row)

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        self.commit_count += 1

    def refresh(self, _row: Any) -> None:
        pass


def _analysis(patches: list[dict[str, Any]]) -> SimpleNamespace:
    return SimpleNamespace(
        id=10,
        user_id=1,
        photo_id=20,
        parsed_result={"acne_patches": patches},
    )


def _patch(patch_id: str, bbox: list[float]) -> dict[str, Any]:
    return {
        "id": patch_id,
        "region": "right_cheek",
        "bbox_norm": bbox,
        "area_ratio": 0.02,
        "coverage": "sparse",
        "dominant_type": "papule",
        "estimated_count": 2,
        "inflammation": "mild",
        "severity": 2,
    }


def test_bbox_center_and_distance() -> None:
    assert tracker._bbox_center([0.2, 0.4, 0.6, 0.8]) == pytest.approx((0.4, 0.6))
    assert tracker._bbox_center([]) == (0.0, 0.0)
    assert tracker._distance((0.0, 0.0), (0.3, 0.4)) == 0.5


def test_two_new_patches_cannot_share_lineage_in_same_analysis(monkeypatch) -> None:
    now = datetime(2026, 7, 13, tzinfo=timezone.utc)
    monkeypatch.setattr(tracker, "_now", lambda: now)
    db = _FakeSession([[], [], []])

    result = tracker.track_patches_for_analysis(
        db,
        _analysis(
            [
                _patch("p1", [0.50, 0.40, 0.60, 0.50]),
                _patch("p2", [0.52, 0.42, 0.62, 0.52]),
            ]
        ),
    )

    snapshots = [row for row in db.added if isinstance(row, PatchLineageSnapshot)]
    assert result.new_lineage_count == 2
    assert result.matched_lineage_count == 0
    assert len({snapshot.lineage_id for snapshot in snapshots}) == 2
    assert db.commit_count == 1


def test_nearby_patch_matches_existing_lineage(monkeypatch) -> None:
    now = datetime(2026, 7, 13, tzinfo=timezone.utc)
    monkeypatch.setattr(tracker, "_now", lambda: now)
    lineage = PatchLineage(
        id=7,
        user_id=1,
        region="right_cheek",
        status="active",
        first_seen_at=now - timedelta(days=3),
        last_seen_at=now - timedelta(days=1),
        snapshot_count=1,
    )
    old_snapshot = PatchLineageSnapshot(
        id=8,
        lineage_id=7,
        analysis_id=9,
        photo_id=19,
        user_id=1,
        patch_id="old",
        region="right_cheek",
        bbox_norm=[0.50, 0.40, 0.60, 0.50],
        area_ratio=0.02,
        coverage="sparse",
        dominant_type="papule",
        estimated_count=2,
        inflammation="mild",
        severity=2,
        created_at=now - timedelta(days=1),
    )
    db = _FakeSession([[lineage], [old_snapshot], [], []])

    result = tracker.track_patches_for_analysis(
        db,
        _analysis([_patch("p1", [0.51, 0.41, 0.61, 0.51])]),
    )

    new_snapshot = next(row for row in db.added if isinstance(row, PatchLineageSnapshot))
    assert result.new_lineage_count == 0
    assert result.matched_lineage_count == 1
    assert new_snapshot.lineage_id == 7
    assert new_snapshot.match_info["matched"] is True
    assert lineage.snapshot_count == 2


def test_empty_analysis_still_advances_lifecycle(monkeypatch) -> None:
    now = datetime(2026, 7, 13, tzinfo=timezone.utc)
    monkeypatch.setattr(tracker, "_now", lambda: now)
    active = PatchLineage(
        id=1,
        user_id=1,
        region="forehead",
        status="active",
        first_seen_at=now - timedelta(days=4),
        last_seen_at=now - timedelta(days=2),
        snapshot_count=1,
    )
    dormant = PatchLineage(
        id=2,
        user_id=1,
        region="chin",
        status="dormant",
        first_seen_at=now - timedelta(days=20),
        last_seen_at=now - timedelta(days=15),
        snapshot_count=1,
    )
    db = _FakeSession([[active], [dormant]])

    result = tracker.track_patches_for_analysis(db, _analysis([]))

    assert result.snapshot_ids == []
    assert active.status == "dormant"
    assert dormant.status == "healed"
    assert db.commit_count == 1