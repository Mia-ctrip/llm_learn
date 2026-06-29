# Skin Care Agent — Backend

AI 皮肤管理产品的后端服务。FastAPI + PostgreSQL + 多模态 LLM Gateway。

> ⚠️ 合规红线：本服务**只描述外观状态**，不诊断病种、不推荐药品、不指导用药。

---

## 当前进度

- [x] 1. 基础脚手架（FastAPI / config / DB session / alembic / `/health`）
- [ ] 2. storage_service + 照片上传
- [ ] 3. ai_service Mock 版（含合规中间件 / 限流）
- [ ] 4. vision 模块（MediaPipe 对齐 + 眼部打码 + 标注图）
- [ ] 5. 接入真实 LLM（Qwen-VL / GLM-4V / MiniMax）
- [ ] 6. vision.tracker —— 跨日单痘追踪（A1）
- [ ] 7. 趋势 / 日记 / 用药 / 问答 + 小程序对接
- [ ] 8. 合规收尾（免责声明 + checkNeedDoctor）

---

## 本地启动

### 1. 依赖

推荐 [uv](https://github.com/astral-sh/uv)（也可用 pip / poetry）。

```powershell
cd backend
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 2. 数据库（PostgreSQL 16）

任选其一：

**A. Docker**
```powershell
docker run -d --name skin-pg `
  -e POSTGRES_USER=skin -e POSTGRES_PASSWORD=skin -e POSTGRES_DB=skin_care `
  -p 5432:5432 postgres:16
```

**B. 本机安装** → 然后建库建用户：
```sql
CREATE USER skin WITH PASSWORD 'skin';
CREATE DATABASE skin_care OWNER skin;
```

### 3. 配置

```powershell
copy .env.example .env
# 编辑 .env，最低限度填好 DATABASE_URL；mvp 第一周 AI_PROVIDER_PRIMARY=mock 即可
```

### 4. 迁移

```powershell
# 当前还没建任何业务表，模型加进来后再生成：
# alembic revision --autogenerate -m "init"
alembic upgrade head
```

### 5. 启动

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. 验证

- `GET http://localhost:8000/health` → `{"status":"ok",...}`
- `GET http://localhost:8000/health/db` → `{"status":"ok","db":"reachable"}`
- 文档：`http://localhost:8000/docs`

---

## 目录约定

详见项目根 `prompt.md` 中确认过的方案。核心拆分：

```
app/
├── api/           # HTTP 路由（薄）
├── services/      # 业务逻辑（厚）
│   ├── ai_service/        # 🌟 统一 AI Gateway（多模型路由 + 合规中间件）
│   ├── storage_service/   # 🌟 对象存储抽象（local / cos）
│   └── vision/            # 🌟 图像预处理 + 跨日追踪 (A1)
├── models/        # SQLAlchemy ORM
├── schemas/       # Pydantic 请求/响应
└── db/            # session + alembic
```
