import type { CheckInViewType } from './check-in-flow.ts';

export type CheckInPhoto = {
  photo_id: number;
  view_type: CheckInViewType;
  width: number | null;
  height: number | null;
  taken_at: string | null;
  quality_status: string | null;
  quality_meta: Record<string, unknown> | null;
  url: string;
  url_expires_at: string;
};

export type CheckIn = {
  check_in_id: number;
  client_request_id: string | null;
  kind: 'quick' | 'standard';
  status: 'draft' | 'complete';
  observed_on: string;
  completed_at: string | null;
  created_at: string;
  diary: Record<string, unknown> | null;
  diary_updated_at: string | null;
  photo_count: number;
  photos: CheckInPhoto[];
};

export type PhotoUpload = {
  photo_id: number;
  client_request_id: string | null;
  check_in_id: number | null;
  view_type: CheckInViewType | null;
  quality_status: string | null;
  quality_meta: Record<string, unknown> | null;
  storage_key: string;
  mime_type: string;
  size_bytes: number;
  width: number | null;
  height: number | null;
  taken_at: string | null;
  url: string;
  url_expires_at: string;
};

export type AuthenticatedRequest = <T>(
  path: string,
  init?: RequestInit,
) => Promise<T>;

type CreateCheckInInput = {
  observedOn: string;
  clientRequestId: string;
};

export type NativePhotoFile = {
  uri: string;
  name: string;
  type: 'image/jpeg';
};

type PhotoUploadInput = {
  file: NativePhotoFile;
  takenAt: string;
  checkInId: number;
  viewType: CheckInViewType;
  clientRequestId: string;
};

export type FormDataLike = {
  append(name: string, value: string | NativePhotoFile): void;
};

export async function createStandardCheckIn(
  request: AuthenticatedRequest,
  input: CreateCheckInInput,
): Promise<CheckIn> {
  return request<CheckIn>('/check-ins', {
    method: 'POST',
    body: JSON.stringify({
      observed_on: input.observedOn,
      kind: 'standard',
      client_request_id: input.clientRequestId,
    }),
  });
}

export function buildPhotoUploadForm(
  input: PhotoUploadInput,
  form: FormDataLike = new FormData() as unknown as FormDataLike,
): FormDataLike {
  form.append('file', input.file);
  form.append('taken_at', input.takenAt);
  form.append('check_in_id', String(input.checkInId));
  form.append('view_type', input.viewType);
  form.append('client_request_id', input.clientRequestId);
  return form;
}

export async function uploadCheckInPhoto(
  request: AuthenticatedRequest,
  form: FormDataLike,
): Promise<PhotoUpload> {
  return request<PhotoUpload>('/photos', {
    method: 'POST',
    body: form as unknown as BodyInit,
  });
}

export async function completeCheckIn(
  request: AuthenticatedRequest,
  checkInId: number,
): Promise<CheckIn> {
  return request<CheckIn>(`/check-ins/${checkInId}/complete`, {
    method: 'POST',
  });
}
