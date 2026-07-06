1. 基础脚手架：FastAPI 起项目、配置、PG 连接、alembic 初始迁移、健康检查端点 → 跑通 GET /health
2. storage_service + 上传：本地实现 + photos 表 + POST /photos → 能上传一张图、能签 URL 读回来
3. ai_service Mock 版：mock provider + 合规中间件 + schema_guard + 限流 → POST /analyses 返回假数据但全链路通
4. vision 模块：MediaPipe 对齐 + 眼部打码 + 标注图渲染（不含 tracker）
5. 接入真实 LLM（Qwen-VL 优先）+ Provider 降级链
6. vision.tracker（A1 核心）：跨日匹配 + lifecycle 状态机 + acne_points 表
7. trends + diary + medications + chat + 小程序对接
8. 首启免责声明 + checkNeedDoctor 规则（合规收尾）


20260627
第 1 步「基础脚手架」完成 ✅

已建立的文件

skin_care_agent/
├── .gitignore                              # 忽略 .env、storage_local、venv
└── backend/
    ├── pyproject.toml                      # 依赖声明（FastAPI/PG/alembic/mediapipe/opencv 等）
    ├── alembic.ini
    ├── .env.example                        # 所有环境变量模板（密钥占位）
    ├── README.md                           # 启动指南 + 进度跟踪
    └── app/
        ├── __init__.py
        ├── main.py                         # FastAPI 入口、CORS、lifespan
        ├── config.py                       # pydantic-settings 读 .env
        ├── api/
        │   └── health.py                   # /health 和 /health/db
        ├── db/
        │   ├── session.py                  # SQLAlchemy engine + get_db 依赖
        │   └── migrations/                 # alembic（env.py + script.py.mako + versions/）
        └── models/
            └── base.py                     # DeclarativeBase + IdMixin + TimestampMixin

设计要点

- config.py 用 pydantic-settings：所有 key 走 .env，硬编码绝迹。fallback providers 解析、本地存储路径解析放进 @property 里，集中收口。
- models/base.py 的 Mixin：IdMixin + TimestampMixin(created_at + deleted_at) 现在就铺好软删的地基，后续每张表 class Photo(Base, IdMixin, TimestampMixin) 即用。
- alembic 的 url 不写在 ini 里，而是 env.py 里从 Settings() 注入——这样 dev/prod 切换不用动两份配置。
- 健康检查分两个端点：/health 单测进程活着，/health/db 单测数据库可达，便于排障。

你需要做的（本地环境）

1. 装好 PostgreSQL 16（Docker 一行或本机安装），按 README 建库建用户
2. copy .env.example .env（dev 阶段 mock 跑通流程，不用任何真实 API key）
3. 装依赖、alembic upgrade head、uvicorn app.main:app --reload
4. 浏览器开 http://localhost:8000/docs，调一下 /health 和 /health/db
