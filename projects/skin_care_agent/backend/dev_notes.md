# Backend 开发日志

> 每完成一个可测试小步，追加一段。三段结构：**我做了什么** / **关键设计决策** / **你需要操作什么**。
> 不是通用文档，是给下一次会话（或未来自己）快速接手用的。
## 项目路线图（初始规划）

以下是项目最初按产品能力拆分的主线。实际实现过程中，Step 3 已从原计划的 Mock 版调整为真实 LLM gateway 接入，Step #6 的 tracker 也提前完成；后续以本文件各章节的实际状态为准。

1. 基础脚手架：FastAPI、配置、PostgreSQL、Alembic、健康检查
2. `storage_service` + 照片上传：本地存储、`photos` 表、签名 URL
3. AI 业务接入：原计划为 Mock 全链路，实际改为真实 LLM、多 Provider 降级、限流、合规和问答
4. vision 模块：MediaPipe 对齐、眼部打码、标注图渲染（不含 tracker）
5. 真实视觉 LLM：Qwen-VL 优先，并接入 Provider 降级链
6. `vision.tracker`：跨日 patch 匹配和生命周期状态机
7. trends、diary、medications、chat 及小程序对接
8. 首启免责声明与 `checkNeedDoctor` 规则（合规收尾）

---

## Step 1 — 项目脚手架

### 我做了什么

搭起最小可运行的 FastAPI + PostgreSQL 骨架。

- `pyproject.toml` — 依赖声明，分核心 / `[dev]` 两组
- `app/config.py` — pydantic-settings 读 `.env`，`@lru_cache` 单例
- `app/db/session.py` — SQLAlchemy `engine` + `SessionLocal` + `get_db()` 依赖
- `app/models/base.py` — `Base` / `IdMixin`（BigInteger 主键）/ `TimestampMixin`（含软删 `deleted_at`）
- `app/main.py` — FastAPI factory + CORS + lifespan
- `app/api/health.py` — `/health`（进程） + `/health/db`（`SELECT 1` 验 PG）

初始项目结构的关键部分：

```text
skin_care_agent/
├── .gitignore
└── backend/
    ├── pyproject.toml
    ├── alembic.ini
    ├── .env.example
    ├── README.md
    └── app/
        ├── main.py
        ├── config.py
        ├── api/health.py
        ├── db/session.py
        ├── db/migrations/
        └── models/base.py
```
- `app/api/health.py` — `/health`（进程） + `/health/db`（`SELECT 1` 验 PG）
- `alembic.ini` + `app/db/migrations/env.py` — DB URL 从 settings 注入，避免硬编码
- `.env.example` — 全部环境变量模板
- `.gitignore` — 忽略 `.venv` / `.env` / `storage_local/`

### 关键设计决策

- **PG 不选 MySQL**：为后面 JSONB（日记 tags）/ 数组+GIN（合规审计）/ pgvector（相似图检索）留路。PG License 永久免费，自己装零成本。
- **`.env` 注入 alembic URL**：alembic.ini 里 `sqlalchemy.url=` 留空，由 `env.py` 从 settings 注入，避免在 git 里漏密钥。
- **软删而非硬删**：所有业务表带 `deleted_at`，配合"不支持用户删数据"的产品定位。
- **uv 管理虚拟环境**：`.venv` 落在项目内（`backend/.venv/`），删项目即删环境。
- **配置全走 pydantic-settings + .env**，不硬编码 key
- **软删除**：所有业务表带 `deleted_at`，查询时 filter `deleted_at IS NULL`
- **user_id 用 BigInteger**：为将来接微信 openid 独立表留空间

### 你需要操作什么

（环境初始化已完成；以下是可复现步骤）

```powershell
cd D:\agent\model\projects\skin_care_agent\backend
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"
copy .env.example .env
```

编辑 `.env`，至少确认：

```text
DATABASE_URL=postgresql+psycopg://skin:skin@localhost:5432/skin_care
```

首次使用前创建 PostgreSQL 16 用户和数据库：

```sql
CREATE USER skin WITH PASSWORD 'skin';
CREATE DATABASE skin_care OWNER skin;
```

然后执行：

```powershell
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

验证：

- `GET http://localhost:8000/health`：进程存活
- `GET http://localhost:8000/health/db`：数据库可达
- `http://localhost:8000/docs`：打开 Swagger 接口页面

---

## Step 2 — storage_service + 照片上传

### 我做了什么

**ORM 模型**：

- `app/models/user.py` — `users` 表（`id` / `wx_openid` / `nickname` / 软删）。微信 openid 独立建模，登录接入前可为空。
- `app/models/photo.py` — `photos` 表（`user_id` 外键 / 唯一 `storage_key` / mime / size / 宽高 / `taken_at` / 软删）。
- `app/models/__init__.py` — 导入新模型，保证 `Base.metadata` 能感知。

**存储抽象** `app/services/storage_service/`：

- `base.py` — `StorageBackend` 抽象类 + `SignedURL` dataclass，定义 `put` / `get` / `exists` / `delete` / `signed_url` 五个方法。
- `local.py` — `LocalStorage` 实现，文件落在 `backend/storage_local/`。
- `signing.py` — HMAC-SHA256 签名/校验，URL 形如 `/files/{key}?exp=...&sig=...`。
- `factory.py` — `get_storage()` 工厂，按 `STORAGE_BACKEND` 返回实现，`@lru_cache` 单例。

**Schema 与 HTTP 路由**：

- `app/schemas/photo.py` — `PhotoUploadResponse` / `PhotoOut`。
- `app/api/photos.py` — `POST /photos` multipart 上传 + `GET /photos/{id}/url` 重签 URL。
- `app/api/files.py` — `GET /files/{key:path}` 验签后返回文件流。

**配置与迁移**：

- `STORAGE_URL_SIGN_SECRET`、`STORAGE_URL_TTL_SECONDS=900`、`UPLOAD_MAX_BYTES=8MB`、`UPLOAD_ALLOWED_MIMES=image/jpeg,image/png,image/webp`。
- `app/db/migrations/versions/0001_init.py` — 手写迁移，建 `users` 和 `photos`，不用 autogenerate。

### 关键设计决策

- **签名 URL 而非直挂静态目录**：防路径猜测 + 短期过期（默认 15 分钟）。
- **接口和 S3/COS 对齐**：`SignedURL` dataclass + `signed_url(key, ttl)`，将来切云存储不改业务。
- **签名内容**：`HMAC(secret, "{key}|{exp}")`；key 防止签名复用到其他文件，exp + 短 TTL 限制 URL 外传后的有效窗口。
- **种子用户 user_id=1**：未接微信登录前，所有请求挂到 id=1；`_ensure_seed_user` 幂等创建。接入 `/auth/wx/login` 后只替换真实登录态，不改表结构。
- **Pillow 二次校验**：不信任 `Content-Type`，用 `PIL.verify()` 确认图像合法并提取宽高。
- **路径结构**：`photos/{uid}/YYYY/MM/DD/{uuid}.{ext}`，按日分目录避免单目录文件爆炸，uuid 防猜测。
- **上传与算法解耦**：上传接口只存盘和入库，不在上传请求中运行 MediaPipe/LLM；分析走后续流程，保持上传接口快速。

### 你需要操作什么

（已完成。`/photos` POST 已能上传，`GET /photos/{id}/url` 已能拿到签名地址。）

验证流程：

1. 执行 `alembic upgrade head`，预期迁移到 `0001_init`，数据库中出现 `users`、`photos` 和 `alembic_version`。
2. 重启 uvicorn，在 `/docs` 中调用 `POST /photos` 上传 jpg/png/webp。
3. 把返回的 `url` 贴到浏览器，确认能看到图片。
4. 修改 URL 末尾的 `sig` 任意一个字符，确认返回 `403 invalid or expired signature`。

示例返回字段：

```json
{
  "photo_id": 1,
  "storage_key": "photos/1/2026/06/29/xxx.jpg",
  "width": 1080,
  "height": 1920,
  "url": "http://localhost:8000/files/photos/1/2026/06/29/xxx.jpg?exp=...&sig=...",
  "url_expires_at": "..."
}
```

---

## Step 3 — ai_service 业务接入（真实 LLM + 合规 + 限流）

### 总体计划

3 拆 4 步交付：

- **3a**（已完成）：限流表 + 中间件 + dev 豁免
- **3b**（本次）：`POST /analyses` + 双表持久化 + 图片压缩后 base64 送 LLM
- **3c**（下一步）：schema_guard 严格版 + 合规出参扫描（违禁词/药品名/诊断词）
- **3d**：`POST /chat` + chat_messages 表

---

### Step 3a — 限流表 + 中间件 + dev 豁免

#### 我做了什么

- `app/models/ai_usage.py` — `AIUsageCounter` 表，`(user_id, kind, usage_date)` 唯一约束。
- `app/db/migrations/versions/0002_ai_usage.py` — 数据库迁移。
- `services/ai_gateway/rate_limit.py` — `try_consume` / `peek` / `require` / `QuotaExceeded`。
- `app/api/ai_debug.py` — `GET /ai/debug/quota` 查看 seed user 当日配额，`POST /ai/debug/quota/{kind}/consume` 手动占额。
- `config.py` 和 `.env.example` — 新增 `AI_RATELIMIT_ENFORCE_IN_DEV`。

#### 关键设计决策

- **单 SQL 完成原子占额**：`INSERT ... ON CONFLICT DO UPDATE ... WHERE count < :limit RETURNING count`。未存在时插入 1；未达上限时更新 +1；达到上限时 WHERE 阻止更新且无 RETURNING，避免“先 SELECT 再 UPDATE”的并发竞态。
- **dev 豁免**：`APP_ENV=dev` 且 `AI_RATELIMIT_ENFORCE_IN_DEV=false`（默认）时直接放行，不查库、不写库，调试时不会误占产线配额。
- **测试真实限流**：将 `.env` 中 `AI_RATELIMIT_ENFORCE_IN_DEV=true`、`AI_ANALYZE_DAILY_LIMIT=2`，连续调用三次应为 200、200、429；测完改回 false。
- **不落在中间件层**：限流是业务动作，让 API handler 显式调用 `rl.require(db, user_id, "analyze")`；健康检查等路径不需要排除清单。

#### 你需要操作什么

1. `.env` 追加（可选，默认就是 false）：

```text
AI_RATELIMIT_ENFORCE_IN_DEV=false
```

2. 执行迁移：

```powershell
cd D:\agent\model\projects\skin_care_agent\backend
.venv\Scripts\activate
alembic upgrade head
```

预期：`Running upgrade 0001_init -> 0002_ai_usage`。

3. 重启 uvicorn。

4. 在 `/docs` 验证：
   - `GET /ai/debug/quota` 应看到 analyze/chat 两条记录，dev 豁免下 `used=0`。
   - `POST /ai/debug/quota/analyze/consume` 应返回 `allowed=true`，dev 豁免下 used 仍为 0。
   - 打开真实限流开关并设置每日上限为 2，连续调用三次应为前两次 200、第三次 429；测完改回 false。

---

### Step 3b — POST /analyses + 双表持久化 + 图片压缩

#### 我做了什么

**新增文件**：
- `models/ai_call_log.py` + `models/analysis.py`
- `db/migrations/versions/0003_analyses.py`（建 ai_call_logs + analyses 两表）
- `services/vision/image_prep.py`（Pillow 压缩 → JPEG q=85 → base64 data URL）
- `services/ai_gateway/prompts.py`（vision system prompt 集中管理 + version 字段）
- `services/analysis_service.py`（业务层：拉图 → 压缩 → gateway → 解析 → 落库）
- `schemas/analysis.py`（AnalyzeRequest / AnalysisOut）
- `api/analyses.py`（`POST /analyses` / `GET /analyses/{id}` / `GET /analyses/by-photo/{photo_id}`）

**修改文件**：
- `models/__init__.py` 追加导入
- `main.py` 挂载 analyses router

#### 关键设计决策

**1. 双表分离（业务表 + 日志表）**
- `analyses` 只存成功结果（前端/趋势查询用）
- `ai_call_logs` 全量落日志（success / llm_failed / parse_failed），供成本核算 + 排障
- 关联：`analyses.ai_call_log_id → ai_call_logs.id`

**2. 幂等（force=false 默认走缓存）**
- 默认命中 `photo_id` 的最近一条成功 analyses 直接返回（`cached=true`）
- `force=true` 时跳过缓存 → 消耗一次配额 → 真调 LLM
- 缓存命中**不消耗配额**（读旧结果不算 AI 调用）

**3. 图片压缩策略**
- 长边 > 1600px → 等比缩放到 1600；JPEG q=85
- **FIXME(step-4)**：接入 vision 模块后应改为"人脸检测 + 裁剪 + 眼部打码"再送 LLM，image_prep 下沉为纯 resize 工具。压缩必然影响细节，只是 MVP 阶段控成本的过渡方案。
- 为什么不上公网 CDN：MVP 本地部署，外网 LLM 拉不到 localhost；上 CDN 是生产方案。

**4. 失败落库分类**
- `llm_failed`：gateway 抛 FatalRequestError / AllProvidersFailedError（HTTP 502）
- `parse_failed`：LLM 返回非 JSON 或 shape 异常（HTTP 422）
- 都写 ai_call_logs 一条，errno + error_message 全存；成功不写 analyses

**5. Prompt 版本化**
- `VISION_ANALYZE_PROMPT_VERSION = "vision-1.0.0"` 写入 `ai_call_logs.input_meta.prompt_version`
- 后面 3c 改 prompt 时递增版本号，就能 SQL 对比"新 prompt 通过率 vs 旧 prompt"

**6. 兼容 gateway 现状**
- 复用 `UnifiedRequest(response_format="json", temperature=0.1)`
- `Message` 中 `image_urls` 塞 base64 data URL；openai_compat 的 `_encode_message` 会自动组装 multipart content
- 不改 gateway 代码

#### 你需要操作什么

**1. 跑迁移**
```bash
cd backend
alembic upgrade head
# 应看到 Running upgrade 0002_ai_usage -> 0003_analyses
```

数据库应新增 `ai_call_logs`、`analyses` 两张表。

**2. 确认 .env 已配 MiniMax key**
```
MINIMAX_API_KEY=sk-xxx（你已配）
MINIMAX_BASE_URL=https://api.minimaxi.com/v1
MINIMAX_MODEL=MiniMax-M3
```

**3. 重启 uvicorn**
```bash
uvicorn app.main:app --reload
```

**4. 端到端验证（在 /docs 或 curl）**

```bash
# a. 先上传一张人脸照片
curl -X POST http://localhost:8000/photos \
     -F "file=@/path/to/face.jpg"
# → 返回 photo_id，记下

# b. 分析这张照片（第一次会真调 LLM）
curl -X POST http://localhost:8000/analyses \
     -H "Content-Type: application/json" \
     -d '{"photo_id": <上一步的id>, "force": false}'
# → 返回完整 parsed_result，cached=false

# c. 再调一次（应走缓存）
curl -X POST http://localhost:8000/analyses \
     -H "Content-Type: application/json" \
     -d '{"photo_id": <同上>, "force": false}'
# → cached=true，秒返回

# d. force 重跑
curl -X POST http://localhost:8000/analyses \
     -d '{"photo_id": <同上>, "force": true}'
# → cached=false，又调一次 LLM

# e. 看历史
curl http://localhost:8000/analyses/by-photo/<photo_id>
# → 返回该照片的所有分析记录（按时间倒序）
```

**5. 观察数据库**
```sql
-- 看每次 AI 调用日志
SELECT id, status, provider, model, input_tokens, output_tokens, latency_ms, error_message
FROM ai_call_logs ORDER BY id DESC LIMIT 10;

-- 看成功的分析结果
SELECT id, photo_id, overall_severity, skin_health_index, needs_doctor, provider
FROM analyses ORDER BY id DESC LIMIT 10;

-- 看 parsed_result 里的痘痘计数
SELECT id, parsed_result->'acne_types' FROM analyses ORDER BY id DESC LIMIT 3;
```

**6. 已知遗留问题（3c 处理）**
- Prompt 里的 schema 是自然语言描述，模型偶尔字段不全时不会拦截
- 违禁词/药品名扫描还没做
- description 字段的合规审计还没做

---

### Step 3b+ — AI 调用可观测性（trace + 每次 provider 落库 + debug API）

#### 我做了什么

**新增文件**：
- `services/ai_gateway/observability.py`
  - `trace_id`（contextvar 存 + `new_trace_id`）
  - `TracedLogger`：结构化日志，格式 `[trace=xxx] event key=val key=val`
  - `sanitize_messages_for_log`：把 base64 data URL 替换成 `<data:image/jpeg;base64,...(N chars)...>`
  - `ProviderCallRecord` dataclass：一次 provider 调用的完整快照
- `db/migrations/versions/0004_ai_call_log_trace.py`

**改造文件**：
- `models/ai_call_log.py`：加 `trace_id / attempt_seq / request_payload` 三列
- `services/ai_gateway/gateway.py`：
  - 新增 `GatewayInvokeResult`（含 response + records）
  - 新增 `invoke_detailed(task, req, *, trace_id, start_attempt_seq, skip_bindings)`
  - `invoke()` 保留旧签名，内部走 `invoke_detailed`
  - 每次 provider 调用/跳过都产生一条 record
  - 关键节点打 trace log
- `services/analysis_service.py`：全部重写
  - 生成 `trace_id` 挂到 contextvar
  - 用 `invoke_detailed` 拿全量 records
  - **B 方案**：gateway 说 ok 但 JSON parse 失败 → 把该 (provider, model) 加入 skip_bindings → 再跑一次 gateway 走下一家；最多 5 轮兜底
  - 每条 record 落一条 `ai_call_logs`
  - `analyses.ai_call_log_id` 关联最终 success 那条
- `api/analyses.py`：response header 加 `X-Trace-Id`；失败 detail 也含 trace_id
- `api/ai_debug.py`：新增
  - `GET /ai/debug/logs?limit=&kind=&status=&provider=`：列最近
  - `GET /ai/debug/logs/{id}`：完整详情
  - `GET /ai/debug/logs/{id}/raw-text`：只看 LLM 原文，方便复制
  - `GET /ai/debug/traces/{trace_id}`：按 trace 聚合，一次业务请求的所有 attempt
- `services/ai_gateway/__init__.py`：导出 `GatewayInvokeResult / ProviderCallRecord / trace_log / new_trace_id / sanitize_messages_for_log`

#### 关键设计决策

**1. 一次业务请求 = 一个 trace_id，一次 provider 调用 = 一条 log**
- 旧版：`/analyses` 一次 → 只落最终一条 log（fallback 时中间的家消失了）
- 新版：minimax parse_failed → glm success → 表里两条 log，同一 trace_id，attempt_seq=1/2
- `analyses.ai_call_log_id` 只指向最终 success 那条，前端看结果不用理会日志

**2. parse 失败的处理归 service，不归 gateway**
- Gateway 只管"网络传输 + 断路"，返回 200 就算 ok
- Service 拿到 response 后 try_parse_json，失败就把这家扔进 `skip_bindings` 再调 gateway
- 好处：gateway 保持职责纯粹；3c 加 schema_guard 时可以复用同一套 skip 机制

**3. request_payload 图片脱敏**
- 直接存 base64 会撑爆 JSONB（180KB 图 × N 条）
- `sanitize_messages_for_log` 把 data URL 换成 `<data:image/jpeg;base64,...(240000 chars)...>` 占位符，长度信息保留
- 排障时能看到 prompt 全文 + 图片元信息，看不到图本身

**4. 兜底解析器 `_extract_json_object`**
- 有些模型（MiniMax M3）习惯在 JSON 前后加散文
- 新 parser 会从文本里抠出第一个平衡的 `{ ... }` 块尝试解析
- 这一步是补救，不是长期方案（3c 会做 schema_guard 严校验）

**5. 保持向后兼容**
- 老的 `gateway.invoke()` 签名不变，别处代码（如 `/ai/debug/invoke`）不需要改
- 内部走 `invoke_detailed`，但只返回 response 或抛异常

#### 你需要操作什么

**1. 跑迁移**
```bash
cd backend
alembic upgrade head
# 应看到 Running upgrade 0003_analyses -> 0004_ai_call_log_trace
```

**2.（可选）智谱 GLM key 现在放上去**
```
# .env
GLM_API_KEY=xxx（你已配）
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-4v-plus
```
factory 会自动把 GLM 加入 `vision_analyze` 的 fallback 链。

**3. 重启 uvicorn**

**4. 端到端验证**

```bash
# 触发一次分析（force=true 跳过缓存）
curl -X POST http://localhost:8000/analyses \
     -H "Content-Type: application/json" \
     -d '{"photo_id": <id>, "force": true}' \
     -i
# 关注 response header 的 X-Trace-Id: xxx
```

**5. Debug API 打点点**

打开 http://localhost:8000/docs，展开 `ai-debug` 标签，重点看：

- `GET /ai/debug/logs?limit=10` → 看最近调用列表，含 status/provider/latency/text_preview
- `GET /ai/debug/logs/{id}` → 完整 request_payload + raw_response
- `GET /ai/debug/logs/{id}/raw-text` → 只看 LLM 原文，方便复制到别处
- `GET /ai/debug/traces/{trace_id}` → 一次 /analyses 的完整 attempt 链（minimax → glm → ...）

**6. 终端看结构化日志**

uvicorn 终端会看到：
```
[trace=abc123] analyze.start photo_id=5 user_id=1
[trace=abc123] analyze.image_prep original=3024x4032 resized=1200x1600 bytes=184320
[trace=abc123] gateway.provider.request seq=1 provider=minimax model=MiniMax-M3 task=vision_analyze
[trace=abc123] gateway.provider.ok seq=1 provider=minimax latency_ms=8321 tokens_in=1500 tokens_out=850 text_preview="The image shows..."
[trace=abc123] analyze.parse.fail provider=minimax text_preview="The image shows..."
[trace=abc123] gateway.provider.request seq=2 provider=glm model=glm-4v-plus
[trace=abc123] gateway.provider.ok seq=2 provider=glm latency_ms=6100 tokens_out=920 text_preview="{\"observation\":..."
[trace=abc123] analyze.parse.ok provider=glm fields=15
[trace=abc123] analyze.done status=success analysis_id=3 log_id=8 provider=glm
```

**7. DBeaver 连接串**

```
Host: localhost
Port: 5432
Database: skin_care
User: skin
Password: skin
```
连上后重点看两张表：`ai_call_logs`（全量日志）+ `analyses`（成功结果）。JSONB 字段 DBeaver 会自动展开。

---

### Step 3b++ — 推理模型 reasoning 分离 + parse_strategy 可观测

#### 我做了什么

**背景**：MiniMax M3 是推理模型，会在正式答案前输出 `<think>...</think>` 块。3b+ 的 `_extract_json_object` 兜底 parser 能从散文里抠出中间的 `{...}` 让业务跑通，但排障时 `text_preview` 显示的是 think 内容，看起来像"输出错乱"。

**新增文件**：
- `services/ai_gateway/parsing.py`：`parse_llm_json(text) -> ParseResult`，返回 `(parsed, reasoning, strategy, stripped_text)`
- `db/migrations/versions/0005_reasoning_fields.py`

**改造文件**：
- `models/ai_call_log.py`：加 `reasoning_text TEXT` + `parse_strategy VARCHAR(16)` 两列
- `services/analysis_service.py`：删除内嵌 parser，改用 `parse_llm_json`；record 落库时并入 `reasoning + strategy`
- `api/ai_debug.py`：
  - `LogRowOut` 加 `parse_strategy` + `has_reasoning`
  - `LogDetailOut` 加 `reasoning_text` + `parse_strategy`
  - `/logs` 和 `/traces/{id}` 支持 `preview_len` query 参数（0-5000，0 表示全文）
  - `/logs/{id}/raw-text` 返回体加 `parse_strategy` + `reasoning_text`
- `services/ai_gateway/prompts.py`：版本升到 `vision-1.2.0`
  - 明确允许 `<think>...</think>` 前置
  - 强调 `</think>` 之后必须只有 JSON
  - 推理块语言不限，正文中文

#### 关键设计决策

**1. parse_strategy 三档：direct / extracted / failed**
- `direct`：模型老实输出 JSON，可以直接 `json.loads`
- `extracted`：从散文里抠出 `{...}` 才成功（MiniMax M3 目前几乎全走这条）
- `failed`：完全没抠到，触发 fallback

意义：将来切模型时，SQL 一句就能出"哪家模型有多少比例是 direct"，量化"结构化能力"。

**2. reasoning 单独存 Text 列，不合并到 raw_response**
- `raw_response` 是 gateway 层原始存档，不该被业务后处理污染
- `reasoning_text` 单独查询、单独展示（未来可能在小程序里"给用户看 AI 是怎么想的"）
- 前端 UI 决策：普通用户看 `parsed_result` + `observation`；高级用户/开发者才展开 `reasoning`

**3. 老数据保留**
- id=1、2、3、4 的 `reasoning_text` 和 `parse_strategy` 都是 NULL
- 保留用于纵向对比"prompt v1.1.0 vs v1.2.0 的 parse_strategy 分布"
- 不清

**4. 兼容非推理模型**
- 没有 `<think>` 标签时 `_extract_reasoning` 返回 `(None, 原文)`
- 后续 GLM-4V / Qwen-VL 直接走 direct 路径，reasoning_text 为空

#### 你需要操作什么

**1. 跑迁移**
```bash
alembic upgrade head
# Running upgrade 0004_ai_call_log_trace -> 0005_reasoning_fields
```

**2. 重启 uvicorn**

**3. 触发一次分析（force=true）**
```bash
curl -X POST http://localhost:8000/analyses \
     -d '{"photo_id": <id>, "force": true}' -i
```

**4. 打点验证**

/docs 里点：

- `GET /ai/debug/logs?limit=5` → 新增 `parse_strategy` 和 `has_reasoning` 字段
  - 应该看到最新一条 `parse_strategy=extracted`（如果 MiniMax 还是 think 模式）或 `direct`（如果它听了新 prompt）
  - `has_reasoning=true` 说明拿到了 think 块
- `GET /ai/debug/logs/{id}` → `reasoning_text` 里能看到完整推理过程
- `GET /ai/debug/logs?preview_len=1000` → 预览拉长到 1000 字符
- `GET /ai/debug/logs/{id}/raw-text` → 返回体多了 `reasoning_text` 和 `parse_strategy`

**5. DBeaver 里跑一句看分布**
```sql
SELECT parse_strategy, COUNT(*)
FROM ai_call_logs
WHERE status = 'success'
GROUP BY parse_strategy;
```

**6. 期望的现象变化**
- 如果 MiniMax M3 遵守新 prompt → `<think>` 里放推理、外面只有 JSON → `parse_strategy=direct`
- 如果 MiniMax 仍然习惯性用散文 → `parse_strategy=extracted`，reasoning 依然被剥离得干干净净
- 两种都能业务成功，区别只是可观测性

---

### Step 3c — schema_guard + 合规出参扫描 + 模板兜底

#### 我做了什么

**新增文件**：
- `services/ai_gateway/schema.py`：pydantic v2 model 定义所有 vision_analyze 字段 + 分级校验
- `services/ai_gateway/compliance.py`：违禁词库（疾病/药品/建议句）+ 扫描 + **B4 模板兜底**
- `services/ai_gateway/validators.py`：一致性 + **needs_doctor 服务端强判**
- `db/migrations/versions/0006_compliance_fields.py`

**改造文件**：
- `models/ai_call_log.py`：加 `schema_errors JSONB` + `compliance_flags JSONB` + `validation_warnings JSONB`
- `services/analysis_service.py`：主循环加两个 fail 分支
  - Parse ok 后跑 schema_guard → 失败也 skip 该 provider + 切 fallback
  - Schema ok 后跑 compliance + validator（不失败，只标记）
  - 最终落 `parsed_result` 用 model.model_dump()（已被模板改写过的合规版本）
- `api/ai_debug.py`：
  - `LogRowOut` 加 `compliance_hit_count / schema_error_count / validation_warning_count / needs_doctor_adjusted`
  - `LogDetailOut` 加 `schema_errors / compliance_flags / validation_warnings`

#### 关键设计决策

**1. 分级 schema（Q1=C）**
- 核心严格：`observation / overall_severity / skin_health_index / needs_doctor / acne_points / acne_types` 缺失或类型不可修复 → schema_failed → 切 fallback
- 边缘宽松：`regions / other_concerns / scars / status_counts` 缺失自动补默认值，未知枚举 fallback 到 unknown
- pydantic v2 `field_validator(mode="before")` 做类型强转（str→int / null→默认值）

**2. B4 模板兜底（Q2）**
- 违禁词命中 → **整字段丢弃**，服务端用模板重生成
- 模板举例：
  - `observation` 命中 → `"共观察到约 X 处皮损特征，主要分布于 Y。"`
  - `regions.{r}.note` 命中 → `"{中文区域}区域可见约 X 处皮损特征。"`
  - `other_concerns.{k}.description` 命中 → 按 severity 拼一句（如 `"该维度呈中度表现。"`）
- 违禁词库分三类：`DISEASE_WORDS`（26 个）/ `DRUG_WORDS`（24 个）/ `ACTION_PATTERNS`（7 个正则）
- 词库集中在 `compliance.py`，合规团队直接编辑列表即可
- 每次命中在 `compliance_flags` 落库：`{field, hits, action, original(前200字), replaced_with}` —— 保留原文用于审计但截断防止表膨胀

**3. needs_doctor 服务端强判（Q3=要）**
- 规则：`overall_severity>=7 OR nodule>0 OR cyst>0 OR broken>=3` → true
- 与 LLM 结果 OR 运算，宁可多提示不可少提示
- 服务端上调时 `validation_warnings.needs_doctor_adjusted=true`，记录原因

**4. schema 失败也切 fallback（用与 parse 相同的 skip_bindings）**
- 复用 3b+ 的 `skip_bindings` 机制，schema 失败也扔进去
- max_parse_retries=5 兜底防死循环

**5. 落库的 parsed_result 是"清洁版"**
- 不再落 LLM 原始 JSON，而是落经过 schema 修正 + 合规重写 + needs_doctor 校准的最终结果
- 想看原文？raw_response.text 仍完整保留
- 用户/前端拿到的永远是合规版本

**6. 老数据兼容**
- id=1~5 的记录 `schema_errors / compliance_flags / validation_warnings` 都是 NULL
- debug API 已处理 None case（compliance_hit_count=0）

#### 你需要操作什么

**1. 跑迁移**
```bash
alembic upgrade head
# Running upgrade 0005_reasoning_fields -> 0006_compliance_fields
```

**2. 重启 uvicorn**

**3. 触发一次分析（force=true）**
```bash
curl -X POST http://localhost:8000/analyses \
     -d '{"photo_id": <id>, "force": true}' -i
```

**4. Debug API 验证**

打开 /docs，看 `GET /ai/debug/logs?limit=5`，最新一条应该有：
- `schema_error_count: 0` （schema 都通过了）
- `compliance_hit_count: N` （0 表示没违规，>0 表示有词被拦下模板改写）
- `validation_warning_count: N`
- `needs_doctor_adjusted: bool`

点 `GET /ai/debug/logs/{id}` 看完整：
- `compliance_flags`：列出被替换的字段和命中的词
- `validation_warnings`：一致性问题 + needs_doctor 校准信息

**5. 端到端测试合规拦截**

想主动触发一次拦截，可以人工构造一次调用（假装 LLM 输出了违规内容）：

方式 A：直接调 `POST /ai/debug/invoke`（跳过 schema/compliance），看模型是否已经会自己不违规
方式 B：临时改 prompt 让 LLM 说"痤疮"，然后跑 /analyses 看 compliance_flags 是否命中

**6. DBeaver 里跑一句看统计**
```sql
-- 各字段被命中的频次
SELECT jsonb_array_elements(compliance_flags)->>'field' AS field,
       COUNT(*)
FROM ai_call_logs
WHERE compliance_flags IS NOT NULL
GROUP BY field
ORDER BY COUNT(*) DESC;

-- 最常见的命中词
SELECT jsonb_array_elements_text(
         jsonb_array_elements(compliance_flags)->'hits'
       ) AS hit_term,
       COUNT(*)
FROM ai_call_logs
WHERE compliance_flags IS NOT NULL
GROUP BY hit_term
ORDER BY COUNT(*) DESC LIMIT 20;
```

**7. 已知局限（不阻塞 MVP）**
- 违禁词库靠字面匹配，"痤 疮"（中间加空格）能绕过——生产建议改为 Aho-Corasick + 正则
- 模板生成的 description 缺少 LLM 观察到的细节（比如"鼻翼旁"这种精细位置）
- 违禁词库需要合规团队定期 review 更新

---

### Step 3d-pre — 痘颗→痘斑（Patch）建模换轨

#### 背景

用户发现"每颗痘单独定位"的设计在重度用户上崩塌：满脸融合成片时无法逐颗计数，跨日追踪无稳定坐标可匹配。

**建模换轨**：
- 主结构：`acne_patch`（一片区域，含 region + bbox + coverage + estimated_count + dominant_type + inflammation + severity + description），**必填**（可为空数组）
- 附加：`acne_point`（单颗痘，含精确定位），**只在轻度可枚举时输出**（<10 颗 且 全为 sparse）

#### 我做了什么

**文档改动**：
- `docs/skin_condition_labels.md`：加 1.4 章"Point vs Patch"，重写 5.1/5.2/5.3（v2 输出格式 + 展示适配 + patch-based 追踪），改 6.2（needs_doctor 加 confluent 触发规则）
- `project_background.md`：核心技术差异点段落改为"区域生命周期追踪 + 单颗痘追踪（轻度可选）"

**代码改动**：
- `schema.py`：新增 `AcnePatch` 类（10 字段，bbox 坐标钳位/coverage/inflammation 枚举校验）；`VisionAnalyzeResult.acne_patches` 必填，`acne_points` 改可选（默认空数组）
- `prompts.py`：升级到 `vision-2.0.0`，重写 system + user prompt 说明 patch-first 规则和 Point 输出条件
- `compliance.py`：
  - 加 `COVERAGE_ZH` / `INFLAMMATION_ZH` 中文映射
  - 加 `_tpl_patch_description`：命中违禁词时用 coverage/type/count/inflammation 拼合规描述
  - `apply_compliance` 主循环加对 `acne_patches[i].description` 的扫描
- `validators.py`：
  - 一致性检查从 `acne_types vs acne_points` 改为 `sum(patch.estimated_count) vs acne_types.total`（±30% 容差）
  - 新加 `point_output_violation` 规则（acne_points > 10 时警告）
  - `_compute_needs_doctor` 加 `任意 patch.coverage == "confluent"` 触发条件
  - `_needs_doctor_reasons` 追加 confluent_patches 原因

**没有 DB 迁移**：`analyses.parsed_result` 是 JSONB，schema 变化不需要 DDL；`ai_call_logs.request_payload.prompt_version` 会自动记为 `vision-2.0.0` 与老数据区分。

#### 关键设计决策

**1. Patch 必填、Point 条件输出**
- v1 用户不确定的问题："这颗痘和上次那颗是同一颗吗？"——重度用户根本没法回答
- v2 用 patch 做主体：即使 estimated_count 抖动，coverage 和 dominant_type 趋势稳定
- Point 只在轻度用户上激活，两条追踪路径并存

**2. bbox_norm 归一化到 [0,1]**
- 相对整张照片而非人脸框：MVP 阶段没有人脸对齐（Task #4 才做）
- 未来 Task #4 上线人脸对齐后，bbox 可以改为相对人脸区域，v3 prompt 升级即可
- 前端画高亮框只需 `img_width * x`

**3. coverage 四档 vs 连续值**
- `sparse/moderate/dense/confluent` 离散化——LLM 更容易稳定输出
- 连续 area_ratio 已经存在，需要更细粒度时可以从这里算

**4. needs_doctor 加 confluent 触发**
- 融合成片临床意义上就是重度信号，不管 severity 数字多少
- 服务端兜底代码里已加，`validation_warnings.reasons` 会记录 `confluent_patches=[p1, p3]`

**5. 一致性容差 ±30%**
- `estimated_count` 本身就是估算，尤其 confluent 情况下
- 强制严等于会频繁误报，反而让日志变噪
- 超过 ±30% 才 warn，代表模型确实自相矛盾

#### 你需要操作什么

**1. 清老数据（Q3=A）**

在 DBeaver 里跑：

```sql
-- 先看当前状态
SELECT COUNT(*) FROM analyses;
SELECT COUNT(*) FROM ai_call_logs;

-- 清 v1 的 analyses 记录（parsed_result 是旧结构）
DELETE FROM analyses;

-- 可选：ai_call_logs 保留（历史 prompt version 对比很有价值）
-- 如果非要清：DELETE FROM ai_call_logs;
```

**2. 重启 uvicorn**

不需要迁移，代码热重载。

**3. 端到端验证**

```bash
# 上传一张照片，force=true 触发真调
curl -X POST http://localhost:8000/analyses \
     -d '{"photo_id": <id>, "force": true}' -i
```

**期望现象**：
- 响应 `parsed_result` 里有 `acne_patches` 数组（可能为空）
- 轻度照片：`acne_points` 也有内容
- 重度照片：`acne_points: []`，`acne_patches` 里有 coverage>=dense 的项

**4. Debug API 检查**

`GET /ai/debug/logs?limit=5`：
- 最新一条 `parse_strategy` 应该是 direct 或 extracted
- `schema_error_count`=0 说明新 schema 通过
- `compliance_hit_count`>0 时说明 patch description 被兜底过

`GET /ai/debug/logs/{id}` 详情看 `parsed_result` 结构。

**5. DBeaver 里看 patch 数据**

```sql
-- 看最新一条 analyses 的 patches
SELECT parsed_result->'acne_patches' FROM analyses ORDER BY id DESC LIMIT 1;

-- 统计新旧 prompt 版本的分布
SELECT input_meta->>'prompt_version' AS version, COUNT(*)
FROM ai_call_logs GROUP BY version ORDER BY version DESC;
```

**6. 已知遗留**
- 前端展示逻辑（sparse→点标 vs dense→高亮框）在小程序阶段处理
- 跨日 patch 匹配算法（Task #6 的 vision.tracker）需要重写：从匈牙利算法+坐标改为 region+特征向量
- 未来 vision 模块（Task #4）上线人脸对齐后，bbox_norm 语义要更新

---

### Step 3d — POST /chat + 合规 + 医疗兜底

#### 我做了什么

**新增文件**：
- `models/chat_message.py`：ChatMessage ORM（问答对业务表）
- `db/migrations/versions/0007_chat_messages.py`
- `schemas/chat.py`：ChatRequest / ChatResponse / ChatMessageOut / ChatContext / ChatHistoryItem
- `services/chat_service.py`：核心业务（医疗兜底 → 拉上下文 → 组装 messages → gateway → 精确删句 → 落库）
- `api/chat.py`：`POST /chat` + `GET /chat/history`

**改造文件**：
- `services/ai_gateway/prompts.py`：加 `CHAT_QA_SYSTEM_PROMPT`（chat-1.0.0）+ `build_chat_context_message`
- `services/ai_gateway/compliance.py`：
  - 加 `MEDICAL_EMERGENCY_KEYWORDS`（20+ 医疗紧急词）+ `detect_medical_emergency`
  - 加 `apply_compliance_to_chat_text`：按中文断句切分，命中违禁词的句子整句删除
  - `MEDICAL_INTERVENTION_MESSAGE`：服务端预设的紧急就医提示文案
- `models/__init__.py`：注册 ChatMessage
- `main.py`：挂载 chat router

#### 关键设计决策（对应 Q1-Q5）

**Q1=B — 前端传上下文**
- `ChatRequest.context.latest_analysis_id`：前端根据当前页面主动传
- 后端拿到 ID 查 `analyses` 表 → `build_chat_context_message` 压缩成短摘要 → 作为额外 system message 注入
- 摘要只包含：observation + severity + skin_health_index + 前 5 个 patches + needs_doctor（防 token 爆炸）

**Q2=A — 单轮（history 可选透传）**
- `ChatRequest.history: list[ChatHistoryItem]` 前端维护多轮
- 服务端零状态，不建 chat_sessions 表
- 需要跨设备同步是 Task #7 之后的事

**Q3=C — 精确删句合规**
- 用 `(?<=[。！？；\.\!\?])\s*|\n+` 中文断句
- 每句独立扫违禁词（复用 vision 那套 DISEASE_WORDS/DRUG_WORDS/ACTION_PATTERNS）
- 命中即整句删，其他句保留 → 用户看到的是自然的短一点的回答
- 兜底：如果所有句子都被删了，返回"抱歉，这个问题涉及无法提供的医疗建议..."的通用回复
- `compliance_flags` 落库：`{field, hits, action:"drop_sentence", original, replaced_with:""}`

**Q4=B — chat_messages 业务表**
- 成功的问答对落 `chat_messages`（干净，前端历史查询用）
- 每次 provider 调用落 `ai_call_logs`（全量，`kind='chat_qa'`，含 fallback/失败）
- 关联：`chat_messages.ai_call_log_id → ai_call_logs.id`
- 医疗兜底的问答也落 `chat_messages`（`medical_intervention=true / provider='server'`），但不落 `ai_call_logs`（根本没调 LLM）

**Q5=B — 医疗关键词兜底**
- `detect_medical_emergency` 命中 → 直接返回 `MEDICAL_INTERVENTION_MESSAGE` 预设文案
- 不调 LLM，成本 0
- 关键词分四类：严重症状（流脓/剧痛/发烧等）、疑似严重疾病（癌/肿瘤）、感染扩散/败血症、特殊风险（怀孕/婴幼儿——涉及用药禁忌）
- 也会占额（防绕限流），落 `chat_messages.medical_intervention=true` 供审计

#### 复用 3b/3b+/3c 基础设施

- Gateway `chat_qa` 链（minimax → deepseek）现成，不改
- 每次 provider 落 `ai_call_logs`：复用 `invoke_detailed`
- trace_id + 结构化日志：复用 observability
- 限流：复用 `rate_limit.require(kind='chat')`
- 合规词库：复用 vision 的 DISEASE_WORDS/DRUG_WORDS/ACTION_PATTERNS

#### 你需要操作什么

**1. 跑迁移**
```bash
alembic upgrade head
# Running upgrade 0006_compliance_fields -> 0007_chat_messages
```

**2. 重启 uvicorn**

**3. 端到端验证**

**场景 A：普通护肤问答**
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "T 区经常出油怎么办"}' -i
```
期望：assistant_message 给出成分级建议（如水杨酸/烟酰胺）；`medical_intervention: false`；X-Trace-Id 有值。

**场景 B：带上下文问答**
```bash
# 先做一次 analysis 拿到 analysis_id，然后：
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "我这个情况严重吗", "context": {"latest_analysis_id": 5}}'
```
期望：回答里能引用你的当前状态（"你右颊有一片中等密度..."）。

**场景 C：医疗紧急兜底**
```bash
curl -X POST http://localhost:8000/chat \
     -d '{"message": "我脸上流脓了怎么办"}'
```
期望：立即返回 `MEDICAL_INTERVENTION_MESSAGE` 全文；`medical_intervention: true`；`provider: "server"`。响应快（不调 LLM）。

**场景 D：多轮追问**
```bash
curl -X POST http://localhost:8000/chat \
     -d '{
       "message": "那两颊呢",
       "history": [
         {"role": "user", "content": "T 区油怎么办"},
         {"role": "assistant", "content": "..."}
       ]
     }'
```
期望：回答理解"两颊"是相对 T 区的追问。

**4. Debug API 检查**
- `GET /ai/debug/logs?kind=chat_qa&limit=5` 看 chat 调用记录
- 医疗兜底的场景**不会**出现在 ai_call_logs（因为没调 LLM）—— 只在 chat_messages 表里
- `GET /chat/history` 看问答历史

**5. DBeaver 里跑一句看统计**
```sql
-- 医疗兜底触发多少次
SELECT medical_intervention, COUNT(*) FROM chat_messages GROUP BY medical_intervention;

-- 合规删句触发多少次
SELECT jsonb_array_length(compliance_flags) AS dropped_sentences, COUNT(*)
FROM chat_messages WHERE compliance_flags IS NOT NULL
GROUP BY dropped_sentences;

-- 各 provider 分布
SELECT provider, COUNT(*) FROM chat_messages GROUP BY provider;
```

**6. 已知遗留**
- 医疗关键词是字面匹配，"化 脓"（加空格）能绕过——生产改 Aho-Corasick
- 中文断句用简单正则，遇到"..."、"—"等复杂标点可能切错——不阻塞 MVP
- 精确删句在极端情况下（每句都命中）会返回兜底提示，可能让用户感到"AI 拒答"

---

## Step 3 主线完成 ✅

- **3a** 限流 + dev 豁免
- **3b** POST /analyses + 双表 + 图片压缩
- **3b+** trace + 每次 provider 落库 + debug API
- **3b++** reasoning 分离 + parse_strategy
- **3c** schema_guard + 合规 + 模板兜底
- **3d-pre** 痘颗 → 痘斑建模换轨（v2.0.0）
- **3d** POST /chat + 医疗兜底 + 精确删句

**下一步方向**（按用户优先级选）：
- Task #4：vision 模块（MediaPipe 对齐 + 眼部打码 + 标注图渲染）
- Task #6：vision.tracker（跨日 patch 匹配）
- Task #7：趋势/日记/用药 + 小程序对接
- Task #8：合规收尾（免责声明 + checkNeedDoctor 前端提示）

---

### Step #6 — vision.tracker（跨日 patch lineage 追踪 + 趋势 API）

#### 我做了什么

**新增文件**：
- `models/patch_lineage.py`：PatchLineage（主线）+ PatchLineageSnapshot（每次分析的快照）
- `db/migrations/versions/0008_lineages.py`
- `services/vision/tracker.py`：匹配算法 + 生命周期状态迁移
- `schemas/lineage.py` + `schemas/trend.py`
- `api/lineages.py`：`GET /lineages` / `GET /lineages/{id}` / `GET /lineages/by-photo/{photo_id}`
- `api/trends.py`：`GET /trends/summary?days=N`

**改造文件**：
- `services/analysis_service.py`：analysis 落库后自动调用 tracker（try/except 包住，tracker 挂不影响 /analyses 200）
- `models/__init__.py`：注册两张新表
- `main.py`：挂载 lineages + trends router

#### 关键设计决策

**1. 一个 region 可有多条 lineage（Q1=B）**
- 右颊今天 2 个 patch → 一个匹配昨天已有 lineage，另一个新建
- 每条 lineage 是"一片持续存在的病灶群"
- 状态 active / dormant（1-14 天没出现）/ healed（>14 天没出现，不再自动接续）

**2. 只做 patch lineage，point 不追踪（Q2=A）**
- point 是"轻度可枚举"的锦上添花，产品价值主要在 patch
- MVP 阶段减少一半工作量
- 未来把 point 当 sparse+1颗的特殊 patch 统一处理即可

**3. 简单匹配算法（Q3=A）**
- 同 region 内 bbox 中心欧氏距离最小的作为候选
- 距离 <= 0.25（归一化后约占面部宽度 25%）→ 匹配
- 否则新建 lineage
- 每条 lineage 一次分析最多匹配一次
- 匹配元信息落 `match_info` JSONB（distance / threshold / reason）供排障

**4. Tracker 挂在 analysis_service 内联跑，不异步**
- MVP 阶段一次分析已经 5-10s，再多 100ms 追踪无所谓
- 失败被 try/except 兜住 → analysis 依然成功返回

**5. 冗余字段设计**
- Snapshot 冗余存 bbox/coverage/type/count/severity
- 避免每次读 lineage 都要 join analyses 表抽 JSONB

**6. 趋势 API 合并两条数据源**
- Analyses → 皮肤指数曲线 + severity 趋势 + 每日总痘数
- Lineages → 活跃/新增/消退区域数 + 分区域摘要
- Highlights 用规则生成人话洞察

#### 你需要操作什么

**1. 跑迁移**
```bash
alembic upgrade head
# Running upgrade 0007_chat_messages -> 0008_lineages
```

**2. 重启 uvicorn**

**3. 验证追踪能力（同照片两次 force=true 分析模拟跨日）**

```bash
curl -X POST http://localhost:8000/analyses -d '{"photo_id": <id>, "force": true}'
curl -X POST http://localhost:8000/analyses -d '{"photo_id": <id>, "force": true}'
curl http://localhost:8000/lineages
# 每个 patch 应该有 2 个 snapshot
```

**4. Debug API**

```bash
curl http://localhost:8000/lineages                        # 列所有
curl "http://localhost:8000/lineages?status=active"        # 筛选
curl "http://localhost:8000/lineages?region=right_cheek"
curl http://localhost:8000/lineages/1                      # 详情 + 时间线
curl http://localhost:8000/lineages/by-photo/<photo_id>
curl "http://localhost:8000/trends/summary?days=30"
```

**5. DBeaver 检查数据**

```sql
SELECT id, region, status, first_seen_at, last_seen_at, snapshot_count
FROM patch_lineages ORDER BY id;

SELECT lineage_id, region, coverage, dominant_type,
       estimated_count, severity, match_info, created_at
FROM patch_lineage_snapshots ORDER BY lineage_id, created_at;

SELECT region, status, COUNT(*) FROM patch_lineages
GROUP BY region, status;
```

**6. 已知遗留 / 未做**
- 每条 lineage 一次分析只能匹配一次；如果两个 patch 都很靠近旧 lineage，只有一个能匹配上 —— 未来上匈牙利分配
- healed 判定用 last_seen_at 时间近似；实际上"用户没拍照" vs "拍了但这个 lineage 没出现"没区分
- 只考虑近 14 天候选；长期用户"两个月前老痘印又发炎"的边缘 case 不 care
- Point 追踪未做（Q2=A）

#### MVP 主线进度更新

到此**后端 API 层面** 5 大 MVP 功能：
- ✅ 拍照记录
- ✅ AI 分析（含跨日追踪 = 护城河）
- ✅ 趋势追踪（`GET /trends/summary`）
- ❌ 痘痘日记（还没做）
- ✅ AI 问答

**剩余后端**：只有痘痘日记未做
**剩余前端**：全部（微信小程序未开始）
**剩余合规**：Task #8 收尾







---

### Step #6 hardening — tracker / trends 稳定性收口

#### 我做了什么

- 修复同一次分析里两个相近新 patch 误共用一条 lineage 的问题。
- 修复 analyses / lineages 的 by-photo 静态路由被动态 ID 路由拦截的问题。
- 空 patch 分析也会推进 dormant / healed 生命周期。
- tracker 异常时显式 rollback，避免污染后续数据库会话。
- tracker、lineage、trend 全链路加入 `view_type`，正面、左侧、右侧不再互相误匹配。
- 趋势区域按“视角 + 面部区域”分组并输出中文视角标签。
- 清理 Ruff 问题并建立首批回归测试。

#### 关键设计决策

- 三个视角使用独立 lineage 空间；同名 region 只有在同视角内才允许匹配。
- 旧照片和旧 lineage 统一标记为 `legacy`，不破坏已有数据。
- tracker 仍是分析后的附属能力；失败记录日志，但分析结果本身保留。

#### 你需要操作什么

（已完成。Ruff 全绿，相关测试已纳入完整测试集。）

---

### Step #7a — 三视角 check-in 基础流程

#### 我做了什么

- 新增 `check_ins` 模型、schema 和 API：创建、列表、详情、完成。
- `standard` 打卡完成前必须包含 `front / left / right`；`quick` 暂不强制照片，为日记入口预留。
- 照片上传支持 `check_in_id + view_type`，同一草稿内同视角重拍会软删除旧照片。
- 照片新增 `quality_status / quality_meta / processed_storage_key`，为下一步质量检测、对齐图预留。
- 新增迁移 `0009_check_ins` 并已在本地 PostgreSQL 成功升级。
- 新增 check-in 与路由测试；当前共 13 个测试通过。

#### 关键设计决策

- “一次记录”作为一等实体，避免把三张照片当成三个互不相关的日记录。
- 标准记录固定三视角，保证后续跨日趋势比较的是同角度照片。
- 兼容旧版单照片上传：不传 check-in 参数时仍按原流程工作。
- 当前只建立质量状态接口，不假装已完成姿态、光照和清晰度检测。

#### 你需要操作什么

- 准备 3–5 组仅用于本地开发的正面、左侧、右侧照片；最好包含一次角度或光线明显不合格的样本。
- 不需要手工迁移数据库；本地库当前已是 `0009_check_ins (head)`。
- 下一开发步：实现拍照质量门槛与标准化对齐，再接痘痘日记字段。
---

## 2026-07-13 — 文档与自主开发记录规则整理 — ✅ 已完成

### 本次完成

- 将 `docs/temp_step.md` 中的初始项目路线图合并到本文件开头。
- 将 `docs/dev_notes.md` 中遗漏的 Step 1 初始化命令、项目目录说明、健康检查验证和 Step 2 上传/签名 URL 验证流程补回对应章节。
- 将 `CLAUDE.md` 改为以本文件作为唯一开发进度来源，并规定每个可验证里程碑完成后立即记录。

### 验证情况

- 已检查 `CLAUDE.md`、本文件开头及 Step 1/Step 2 章节。
- 本次只修改文档规则和开发日志，没有运行代码测试。

### 当前阻塞或遗留

- 无。
- `docs/dev_notes.md` 和 `docs/temp_step.md` 保留为历史/补充文档，不再作为最新进度来源。

### 下一步

- 下一次自主开发开始前读取 `project_background.md`、本文件和 `CLAUDE.md`。
- 完成每个独立可验证里程碑后，立即在本文件追加记录。
---

## 2026-07-13 — Step 1/2/3a 文档完整性补充 — ✅ 已完成

### 本次完成

- 依照 `docs/dev_notes.md` 补全 Step 1 的项目结构、PostgreSQL 初始化和健康检查上下文。
- 补全 Step 2 的 ORM 模型、存储抽象、配置、迁移、签名规则、Pillow 校验和上传验证流程。
- 补全 Step 3a 的文件清单、原子限流机制、dev 豁免、真实限流测试和操作步骤。

### 验证情况

- 已逐项对照 `docs/dev_notes.md` 的 Step 1、Step 2、Step 3a 章节。
- 已检查补充内容的章节顺序和 Markdown 结构。
- 本次只修改日志文档，没有运行代码测试。

### 当前阻塞或遗留

- 无。

### 下一步

- 后续开发继续只在本文件追加每个可验证里程碑的完整记录。
---

## 2026-07-14 — Step #7b 拍照质量门槛与几何标准化 — ✅ 已完成

### 本次完成

- 新增 `app/services/vision/quality.py`：使用本地 MediaPipe Face Landmarker 检查分辨率、清晰度、光照、完整人脸、多人脸、头部倾斜及 front/left/right 视角。
- `POST /photos` 在标准 check-in 上传时执行质量门槛；失败返回 422、错误码、中文重拍提示和可排障指标，失败照片不进入存储与数据库。
- 新增 `app/services/vision/normalization.py`：合格照片保留原图，同时生成 `1024×1280` JPEG 几何标准化副本；不美白、不调色、不锐化、不修改皮肤内容。
- `photos.processed_storage_key` 正式启用；AI 分析优先读取标准化副本，旧照片仍回退到原图。
- EXIF 方向在上传尺寸读取和 LLM 图片准备阶段统一纠正。
- 新增 `scripts/download_face_landmarker.ps1`，固定官方模型 URL 和 SHA256；`backend/model_assets/` 已加入 Git 忽略。
- 应用退出时关闭缓存的 Face Landmarker，释放本地模型资源。

### 验证情况

- 私有样本校准：`set01/front.jpg`、`left.jpg`、`right.jpg` 分别按 front/left/right 全部通过。
- `set02` 与 `set03` 共 6 张下半脸、裁切或不完整构图样本全部被拒绝；原因覆盖 `image_blurry`、`face_cut_off`、`face_not_detected`。
- 已目视检查三张标准化输出：额头、两颊和下巴保留，左右侧没有镜像，输出尺寸均为 `1024×1280`。
- Ruff：全绿。
- Pytest：20 passed，1 条既有 Starlette/httpx2 弃用警告。
- 模型下载脚本已通过 PowerShell 语法解析；本地模型 SHA256 为 `64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF`。

### 当前阻塞或遗留

- MediaPipe 官方说明输入图片在设备本地处理、不会发送到 Google，但 Tasks API 会发送性能和使用指标。当前受限环境中的遥测连接失败；生产上线前需在隐私政策中披露，或将预处理进程部署到禁止外连的网络环境。官方说明：https://github.com/google-ai-edge/mediapipe#privacy-notice
- 当前完成的是后端拍后校验；微信小程序相机中的实时参考框和即时姿态提示尚未实现。
- 眼部遮盖尚未实现；需要在不遮挡眼周皮肤观察区域的前提下单独设计。

### 下一步

- 设计并实现痘痘日记字段，使 quick/standard check-in 都能记录睡眠、压力、饮食、经期和护肤变化。
- 随后按 check-in 而不是单张照片聚合三视角分析和趋势，避免同日多图重复计数。
- 用户当前无需操作；若在新环境重新拉取项目，运行 `backend/scripts/download_face_landmarker.ps1`。

---

## 2026-07-14 — Step #7c 痘痘日记字段与接口 — ✅ 已完成

### 本次完成

- `check_ins` 新增 `diary_data JSONB` 和 `diary_updated_at`，日记与 quick/standard check-in 一对一关联。
- 新增结构化日记 schema，覆盖睡眠时长与质量、压力等级、经期阶段、饮食标签、护肤变化、新护肤品、用户主动填写的外用产品及备注。
- 数值范围、枚举、字符串长度、额外字段均由 Pydantic 严格校验；标签和产品名称会去重，产品名称会去除首尾空白。
- `POST /check-ins` 支持可选 `diary`；新增 `PUT /check-ins/{id}/diary` 完整替换接口，传空对象可清空日记。
- 已完成的 check-in 仍允许修正日记，但不会改变照片、完成状态或完成时间。
- `CheckInOut`、列表和详情接口统一返回 `diary` 与 `diary_updated_at`。
- 新增迁移 `0010_check_in_diary`，包含 JSONB 对象类型数据库约束，并已在本地 PostgreSQL 从 `0009_check_ins` 升级至 head。
- README 和项目状态已同步更新。

### 验证情况

- 日记与 OpenAPI 针对性测试：10 passed。
- 完整回归：24 passed；仅保留 1 条既有 Starlette/httpx2 弃用警告。
- Ruff 全量静态检查通过；本次涉及的 Python 文件已通过 Ruff format。
- Alembic：升级前为 `0009_check_ins`，升级后为 `0010_check_in_diary (head)`。
- 数据库反射确认 `diary_data JSONB`、`diary_updated_at` 和 `ck_check_ins_diary_data_object` 均已创建。
- 真实 FastAPI + PostgreSQL 往返通过：创建 quick check-in 并写日记、完整替换日记、完成 check-in、完成后修正日记；临时记录已清理。

### 当前阻塞或遗留

- 本子步骤无阻塞。
- 当前仅完成日记采集与持久化，尚未把三视角分析和日记按 check-in 聚合，也尚未生成生活因素关联提示。

### 下一步

- 按 check-in 聚合 front/left/right 三视角分析，定义单次记录的皮肤指数、严重度、数量和就医提示合并规则。
- 将趋势 API 从按单张照片统计收口为按 check-in 统计，避免同一天三张照片被重复计数。
- 聚合稳定后开始微信小程序相机参考框、日记表单和趋势页面。

---

## 2026-07-14 — Step #7d 三视角分析聚合与趋势收口 — ✅ 已完成

### 本次完成

- 新增 `app/services/check_in_aggregation.py`，批量读取 check-in 有效照片的最新成功分析，避免 force 重跑记录重复参与统计。
- 新增 `GET /check-ins/{id}/analysis-summary`，返回 `empty / partial / ready`、缺失照片视角、缺失分析视角、分视角明细及聚合指标。
- 聚合规则固定为：整体严重度取最高值、皮肤指数取平均值、`needs_doctor` 做 OR；每个视角先按 region 累加 patch 数量，再对重叠视角的同一 region 取最大值。
- 聚合响应同时返回日记数据，为后续生活因素关联提供同一 check-in 上下文。
- `GET /trends/summary` 改为只使用已完成且聚合为 `ready` 的 check-in；同一天只选一条并优先 standard，避免三张照片或多次记录形成重复曲线点。
- 趋势响应新增 `source / check_in_id / total_check_ins / incomplete_check_ins / superseded_check_ins / total_legacy_records`，前端可以区分聚合记录、旧数据和不完整记录。
- 保留旧版无 check-in 照片：记录日期优先取 `taken_at`，否则取照片创建时间；同一照片多次分析只取最新成功结果，同一天旧照片只保留最新一张。
- 本步骤采用读取时聚合，没有新增数据库表或迁移；数据库 head 仍为 `0010_check_in_diary`。
- README 和项目状态已同步更新。

### 验证情况

- 新增三视角区域去重、严重度/指数/就医提示合并、缺失分析视角、旧结构计数回退和同日 standard 优先测试。
- 聚合、趋势与 OpenAPI 针对性测试：17 passed。
- 完整回归：28 passed；仅保留 1 条既有 Starlette/httpx2 弃用警告。
- Ruff 全量静态检查通过；本次涉及的 8 个 Python 文件已通过 Ruff format 检查。
- 真实 FastAPI + PostgreSQL 往返通过：临时写入一个完整三视角 check-in 和三条分析，聚合结果为严重度 5、皮肤指数 70、区域去重总数 15、`needs_doctor=true`；`days=1` 趋势只生成一个 check-in 点并计入 3 条底层视角分析。
- 上述临时 check-in、3 张照片和 3 条分析均已精确清理。
- 现有旧数据验证：同一旧照片的两次分析在趋势中只保留最新一条，`total_analyses` 从重复的 2 收口为 1。

### 当前阻塞或遗留

- 本子步骤无阻塞。
- `jaw / temple` 当前 region 标签没有左右侧语义；跨视角取最大值能稳定避免重复，但可能低估左右两侧同时存在的独立数量。需要更精确时应先升级视觉 schema，而不是在聚合层猜测。
- 当前聚合为读取时计算，适合 MVP 数据规模；数据量明显增长后再评估物化汇总表或缓存。
- 微信小程序尚未开始，用户还不能从前端完成拍摄、日记、分析与趋势闭环。

### 下一步

- 创建微信原生小程序工程骨架和 API client，先跑通开发环境连接。
- 实现首页、standard/quick check-in 流程、三视角相机参考框、日记表单和聚合结果页。
- 接入趋势页面，并对 `partial / incomplete / superseded / legacy` 状态提供用户可理解的提示。

---

## 2026-07-15 — Step #6 correctness：Check-in 感知的生命周期收口 — ✅ 已完成

### 本次完成

- 将 lineage 从“按服务器时间自动老化”改为“由同视角有效照片的观察证据推进”。
- 新增 `patch_lineage_observations`，逐条记录 `present / missing`、观察日期、check-in、照片、分析、是否推进状态及原因。
- `patch_lineages` 新增首次出现日、最后出现日、最后观察日、最后出现 check-in、连续缺失次数和状态原因；snapshot 新增 `check_in_id / observed_on`。
- `photos` 新增 tracker 分析标记和时间；同一照片即使强制重分析，也只允许推进一次。
- 完成状态机：`present → active`；第一次有效 `missing → dormant`；至少连续两次有效 missing 且距最后 present 满 14 天才 `healed`。
- 没上传、缺少该视角、check-in 尚未完成或分析尚未成功时不生成 missing，也不改变 lineage。
- 移除固定 14 天匹配窗口：中间没有观察时，相隔超过 14 天仍可按 region + bbox 继续尚未 healed 的原链。
- 草稿照片先分析时延迟追踪；check-in 完成时原子处理已有分析。check-in 完成后才分析的照片会在分析成功后立即追踪。
- 新增 `GET /lineages/by-check-in/{check_in_id}`；lineage 详情返回观察证据时间线，列表和趋势改用观察日期排序与统计。
- 新增迁移 `0011_check_in_lineages`。历史 snapshot 回填为 present；旧墙上时间推导的 dormant/healed 不继承，统一按最后一张历史照片恢复为 active；已有成功分析照片做幂等标记，历史草稿 check-in 保留完成时首次追踪能力。

### 关键设计决策

- “用户没拍”是没有观察，不等于病灶消失；只有同一视角、质量通过且分析成功的照片才能提供 missing 证据。
- front / left / right 的 lineage 空间继续隔离，某次 check-in 缺少 right 时不会影响 right lineage。
- 同一观察日不会重复累计 missing；历史日期的迟到分析保留审计记录，但不倒推当前状态。
- 尚未 healed 的同位置病灶可以跨长时间无照片间隔接续；一旦由重复缺失证据 healed，后续相似病灶新建 lineage。
- check-in 完成与已有分析的生命周期写入放在同一事务；失败会回滚完成状态，允许客户端重试。

### 验证情况

- Alembic 已从 `0010_check_in_diary` 升级到 `0011_check_in_lineages (head)`，并完成一次 downgrade → upgrade 往返。
- 数据库反射确认 observation 表、日期字段、照片幂等字段、外键、索引和 `lineage_id + photo_id` 唯一约束均存在。
- 真实 PostgreSQL 链路通过：6 月 1 日 right present 为 active；6 月 10 日只有 front 照片时 right 仍 active；随后 right 首次 missing 为 dormant；6 月 20 日第二次 missing 后为 healed。
- 上述链路生成且仅生成 `present / missing / missing` 三条证据；同一照片重分析返回 `photo_already_tracked`，观察数不增加。
- 真实 FastAPI 验证 `GET /lineages/{id}` 与 `GET /lineages/by-check-in/{id}` 均返回正确状态和时间线；临时 check-in、照片、分析、snapshot、observation 和 lineage 已精确清理。
- 完整回归：33 passed；仅保留 1 条既有 Starlette/httpx2 弃用警告。
- Ruff 全量静态检查通过，Git diff whitespace 检查通过。

### 当前阻塞或遗留

- 当前仍是 region + bbox 中心距离的贪心匹配；病灶密集、合并或分裂时可能发生身份交换，后续可升级为全局分配与更稳定的视觉特征。
- 同日多次 check-in 会保留审计记录，但只有向前的观察日累计 missing；产品层仍应引导每天完成一个主 check-in。
- 微信小程序尚未展示生命周期链，也没有周期提醒、局部区域对照 UI。

### 下一步

- 创建微信原生小程序骨架，先接入 check-in、三视角拍摄、日记、分析聚合和趋势页面。
- 在结果页加入按 check-in 查询的区域生命周期入口，并用用户可理解的文案区分“仍可见、一次未见、连续未见”。
- 用户当前无需操作；其他环境部署时运行 `alembic upgrade head`。

---

## 2026-07-16 — MVP 客户端平台策略 — ✅ 已完成

### 本次完成

- 评估 MVP 阶段使用微信小程序还是 App。
- 确定直接以 App 作为主产品进行 MVP 内测和首次上线准备。
- 确定小程序作为后续获客和低门槛入口，而不是先做小程序再迁移 App。
- 将完整分析记录到 `docs/platform_strategy.md`。

### 关键决策

- 先做 App MVP，范围聚焦注册、三视角拍照、AI 分析、日记和趋势闭环。
- 先进行 20～50 人小范围内测，验证 7～14 天行为，再公开上架。
- 后端用户 ID、照片、check-in、日记、分析和趋势数据保持客户端无关，避免未来平台迁移。

### 当前阻塞或遗留

- App 技术栈尚未确定。
- 真实用户体系、App 前端和上线备案仍未实现。

### 下一步

- 设计 App MVP 信息架构和首条用户主流程。
- 评估 Flutter、React Native 或 Expo，并创建客户端工程骨架。