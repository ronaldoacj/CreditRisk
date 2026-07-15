"""
DataPipeline/config.py -- wrapper fino sobre common/config_base.py.
Mantem compatibilidade 100% com `from config import X` usado nos scripts
originais. NAO DUPLIQUE valores aqui -- edite sempre em common/config_base.py.
"""
from common.config_base import *  # noqa: F401,F403
