import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CHECK_IN_VIEWS,
  createClientRequestId,
  localObservedOn,
  nextIncompleteView,
  qualityFailureMessage,
} from '../src/lib/check-in-flow.ts';

test('standard check-in uses front, left, right order', () => {
  assert.deepEqual(
    CHECK_IN_VIEWS.map((view) => view.type),
    ['front', 'left', 'right'],
  );
});

test('nextIncompleteView returns the first missing view', () => {
  assert.equal(nextIncompleteView([]), 'front');
  assert.equal(nextIncompleteView(['front']), 'left');
  assert.equal(nextIncompleteView(['front', 'left']), 'right');
  assert.equal(nextIncompleteView(['front', 'left', 'right']), null);
});

test('nextIncompleteView ignores capture order and duplicate views', () => {
  assert.equal(nextIncompleteView(['right', 'front', 'right']), 'left');
});

test('qualityFailureMessage translates backend quality errors', () => {
  assert.equal(qualityFailureMessage('face_not_detected'), '没有检测到完整人脸，请正对参考框重拍。');
  assert.equal(qualityFailureMessage('image_blurry'), '照片不够清晰，请保持手机稳定后重拍。');
  assert.equal(qualityFailureMessage('unknown_code'), '照片未通过质量检查，请按参考框重拍。');
});
test('createClientRequestId creates an RFC 4122 version 4 UUID', () => {
  assert.equal(
    createClientRequestId(() => 0),
    '00000000-0000-4000-8000-000000000000',
  );
  assert.match(
    createClientRequestId(),
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
  );
});

test('localObservedOn uses the device local calendar date', () => {
  assert.equal(localObservedOn(new Date(2026, 6, 5, 23, 59)), '2026-07-05');
});
