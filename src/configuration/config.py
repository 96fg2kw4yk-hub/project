import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.parent  # 往上找三个目录

DATA_DIR = ROOT_DIR / 'data'

NER_DIR = 'ner'

RAW_DATA_DIR = DATA_DIR / NER_DIR / 'raw'

PROCESSED_DATA_DIR = DATA_DIR / NER_DIR / 'processed'

LOG_DIR = ROOT_DIR / 'logs'

CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'

# 定义各类文件的地址
RAW_DATA_FILE = str(RAW_DATA_DIR / 'data.json')

MODEL_NAME = 'google-bert/bert-base-chinese'

# 超参数
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5

SAVE_STEPS = 20

# NER任务分类标签
LABELS = ['B', 'I', 'O']

# 数据库连接（从环境变量读取）
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'port': int(os.getenv('MYSQL_PORT', '3306')),
    'database': os.getenv('MYSQL_DATABASE', 'gmall')
}

NEO4J_config = {
    'uri': os.getenv('NEO4J_URI', ''),
    'auth': (
        os.getenv('NEO4J_USER', 'neo4j'),
        os.getenv('NEO4J_PASSWORD', '')
    )
}

API_KEY = os.getenv('DEEPSEEK_API_KEY', '')

WEB_STATIC_DIR = ROOT_DIR / 'src' / 'web' / 'static'