"""
Configuration file for EVE model files and download URLs.
Update these URLs when you have the files hosted on a server.
"""

# Model file URLs (update these with actual URLs when available)
MODEL_URLS = {
    'peft': {
        'url': 'https://example.com/eve_peft_model.zip',  # Placeholder
        'filename': 'peft_model.zip',
        'extract': True,
        'description': 'PEFT adapter weights'
    },
    'train_db': {
        'url': 'https://example.com/eve_train_db.tar.gz',  # Placeholder
        'filename': 'train_db.tar.gz',
        'extract': True,
        'description': 'Training database for foldseek'
    },
    'train_csv': {
        'url': 'https://example.com/eve_train_csv.csv',  # Placeholder
        'filename': 'train_csv.csv',
        'extract': False,
        'description': 'Training CSV file'
    }
}



# Default cache directory
DEFAULT_CACHE_DIR = '~/.eve_cache'

