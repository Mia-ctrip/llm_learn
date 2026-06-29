# Skin Care Agent — Backend

AI 皮肤管理产品的后端服务。FastAPI + PostgreSQL + 多模态 LLM Gateway。

> ⚠️ **合规红线**：本服务**只描述外观状态**，不诊断病种、不推荐药品、不指导用药。

---

## 这是什么

长期皮肤（痘痘）追踪 + 趋势可视化的后端。**不是诊断工具**。核心能力：

- 📷 拍照上传 + 短期签名 URL（对齐云存储 Presigned URL 模式）
- 🧠 AI 状态描述（多模型 Gateway：Qwen-VL / GLM-4V / MiniMax / 豆包，含降级链 + 合规中间件 + 限流）
- 🔍 跨日单痘追踪（MediaPipe 对齐 + 匈牙利匹配 + 状态机，**产品护城河**）
- 📈 趋势可视化、痘痘日记、用药日志（仅记录，不指导用药）、AI 问答

---

## 快速开始

### 1. 准备 Python 环境

需要 Python ≥ 3.11，推荐 [uv](https://github.com/astral-sh/uv)。

```powershell
cd backend
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 2. 准备 PostgreSQL 16

**A. Docker**
```powershell
docker run -d --name skin-pg `
  -e POSTGRES_USER=skin -e POSTGRES_PASSWORD=skin -e POSTGRES_DB=skin_care `
  -p 5432:5432 postgres:16
```

**B. 本机安装** → 建库建用户：
```sql
CREATE USER skin WITH PASSWORD 'skin';
CREATE DATABASE skin_care OWNER skin;
```

### 3. 配置 `.env`

```powershell
copy .env.example .env
```

最低限度填好：
- `DATABASE_URL`（按上面建的库/用户/密码）
- `STORAGE_URL_SIGN_SECRET`（dev 用默认值即可，生产改强随机串）
- `AI_PROVIDER_PRIMARY=mock`（MVP 第一周不接真实 LLM）

### 4. 跑迁移

```powershell
alembic upgrade head
```

### 5. 启动

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. 验证

| 检查 | URL | 期望 |
|---|---|---|
| 进程 | `GET /health` | `{"status":"ok",...}` |
| 数据库 | `GET /health/db` | `{"status":"ok","db":"reachable"}` |
| 接口文档 | http://localhost:8000/docs | Swagger UI 打开 |

---

## 当前能力

| 模块 | 状态 | 接口 |
|---|---|---|
| 健康检查 | ✅ | `GET /health` · `GET /health/db` |
| 照片上传 | ✅ | `POST /photos` · `GET /photos/{id}/url` · `GET /files/{key}` |
| AI 分析 | 🚧 | 待实现 |
| 跨日追踪 | 🚧 | 待实现 |

完整开发进度与设计决策见 **[`dev_notes.md`](./dev_notes.md)**。

---

## 目录结构

```
app/
├── api/              # HTTP 路由（薄，只做参数校验和编排）
│   ├── health.py
│   ├── photos.py
│   └── files.py
├── services/         # 业务逻辑（厚）
│   ├── ai_service/        # 统一 AI Gateway（多模型路由 + 合规中间件）  🚧
│   ├── storage_service/   # 对象存储抽象（local / cos）                 ✅
│   └── vision/            # 图像预处理 + 跨日追踪 (A1)                  🚧
├── models/           # SQLAlchemy ORM
├── schemas/          # Pydantic 请求/响应
├── db/
│   ├── session.py
│   └── migrations/   # Alembic
├── config.py         # pydantic-settings 读 .env
└── main.py           # FastAPI factory
```

---

## 常用命令速查

```powershell
# 进虚拟环境
.venv\Scripts\activate

# 启动（reload 模式）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 新建迁移
alembic revision -m "describe change"            # 手写
alembic revision --autogenerate -m "..."         # 自动生成（仅参考）

# 应用迁移 / 回滚
alembic upgrade head
alembic downgrade -1

# 进数据库
psql -h localhost -U skin -d skin_care
```

---

## 设计原则

- **MVP 不过度工程化**：不上 K8s、不上分布式、不上 Milvus。本地能跑、能验证产品假设即可。
- **不与微信耦合**：后续要迁 App，登录/通知层保持抽象。
- **密钥全部走 `.env`**：不进 git，`.env.example` 是模板。
- **抽象只在边界**：存储、AI Provider、登录 —— 这些必然要换实现的地方做抽象；业务代码不做提前设计。
