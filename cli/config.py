"""
Configuration file for EVE model files and download URLs.
Update these URLs when you have the files hosted on a server.
"""

# Model file URLs (update these with actual URLs when available)
MODEL_URLS = {
    'eve_data': {
        'url': 'https://zenodo.org/records/16915527/files/EVE_DATA.zip?download=1',
        'filename': 'EVE_DATA.zip',
        'extract': True,
        'description': 'EVE data archive containing all required files'
    }
}

# File checksums for verification (optional)
FILE_CHECKSUMS = {
    'EVE_DATA.zip': None  # No checksum provided, can be added later if needed
}

# Default cache directory
DEFAULT_CACHE_DIR = '~/.eve_cache'

