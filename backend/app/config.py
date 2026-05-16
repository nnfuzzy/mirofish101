"""
配置管理
统一从项目根目录的 .env 文件加载配置
"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False
    
    # ── LLM configuration ────────────────────────────────────────────
    # New (preferred): LiteLLM-style provider/model strings. LiteLLM reads
    # provider keys (GEMINI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY,
    # MISTRAL_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, …) directly from os.environ.
    #
    # Examples:
    #   LITELLM_MODEL=gemini/gemini-2.5-flash
    #   LITELLM_MODEL=anthropic/claude-haiku-4-5
    #   LITELLM_MODEL=deepseek/deepseek-chat
    #   LITELLM_MODEL=mistral/mistral-small-latest
    #   LITELLM_MODEL=groq/llama-3.3-70b-versatile
    #   LITELLM_MODEL=openai/gpt-5.4
    LITELLM_MODEL = os.environ.get('LITELLM_MODEL')
    LITELLM_REPORT_MODEL = os.environ.get('LITELLM_REPORT_MODEL') or LITELLM_MODEL

    # Legacy (still supported): OpenAI-compatible client config. If
    # LITELLM_MODEL is unset, the client falls back to these values and
    # routes everything through the OpenAI SDK as before.
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # Zep配置 — prefer the project-prefixed key so this fork can run
    # alongside other Zep-using apps that share an env file.
    ZEP_API_KEY = os.environ.get('MIROFISH101_ZEP_API_KEY') or os.environ.get('ZEP_API_KEY')
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls):
        """验证必要配置 / Validate required configuration."""
        errors = []
        # Either the LiteLLM path (preferred) or the legacy OpenAI-style
        # path must be configured. LiteLLM provider keys themselves are
        # checked at call time by litellm based on the model prefix.
        if not cls.LITELLM_MODEL and not cls.LLM_API_KEY:
            errors.append(
                "Neither LITELLM_MODEL nor LLM_API_KEY is configured. "
                "Set LITELLM_MODEL=<provider>/<model> (e.g. gemini/gemini-2.5-flash) "
                "and the matching provider key, or set LLM_API_KEY for the legacy OpenAI-compatible client."
            )
        if not cls.ZEP_API_KEY:
            errors.append("ZEP_API_KEY 未配置")
        return errors

