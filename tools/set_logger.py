from datetime import datetime
import logging.config
import os

def setupLog(ident, level='DEBUG', handlers_type='console'):
    """
    Generate system log with either console or file handler.

    Parameters:
    ident (str): Identifier for the logger.
    level (str): Logging level ('DEBUG', 'INFO', 'WARN', 'ERROR').
    handler_type (str): Type of handler to use ('console' or 'file').

    Returns:
    logging.Logger: Configured logger.
    """

    assert level in ('DEBUG', 'INFO', 'WARN', 'ERROR'), "Invalid logging level" # DEBUG > INFO > WARN > ERROR
    LOG_CONF = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            ident: {
                'handlers': [handlers_type],
                'level': level,
                'propagate': False
            }
        },
    }

    if handlers_type == 'file':
        log_dir = os.path.join(os.getcwd(), 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        DEBUG_FILE = os.path.join(log_dir,f'{ident}_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
        
        LOG_CONF['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'standard',
            'filename': DEBUG_FILE,
            'mode': 'w+',
            'maxBytes': 1024 * 1024 * 200,  # 200 MB
            'backupCount': 9,
            'encoding': 'UTF-8'
        }

    logging.config.dictConfig(LOG_CONF)
    return logging.getLogger(ident)

logger = setupLog(ident='_', level='INFO', handlers_type='console')