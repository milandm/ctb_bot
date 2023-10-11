"""
Django settings for api_crud project.

Generated by 'django-admin startproject' using Django 2.0.6.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.0/ref/settings/
"""
import os
from dotenv import load_dotenv
import datetime


load_dotenv()


REACT_APP_CONTENTFUL_API_KEY=os.getenv('REACT_APP_CONTENTFUL_API_KEY')
REACT_APP_CONTENTFUL_GRAPHQL_URL=os.getenv('REACT_APP_CONTENTFUL_GRAPHQL_URL')
REACT_APP_CONTENTFUL_SPACE_ID=os.getenv('REACT_APP_CONTENTFUL_SPACE_ID')
REACT_APP_OKTA_CLIENT_ID=os.getenv('REACT_APP_OKTA_CLIENT_ID')
REACT_APP_OKTA_SERVER_URL=os.getenv('REACT_APP_OKTA_SERVER_URL')
REACT_APP_PREVIEW_CONTENTFUL_API_KEY=os.getenv('REACT_APP_PREVIEW_CONTENTFUL_API_KEY')
REACT_APP_TAX3PO_API_KEY= os.getenv('REACT_APP_TAX3PO_API_KEY')
REACT_APP_TAX3PO_API_DEV_URL= os.getenv('REACT_APP_TAX3PO_API_DEV_URL')

TAX3PO_API_ENDPOINT_CONVERSATION="/conversation"
TAX3PO_API_ENDPOINT_STRUCTURED_BUFFER="/conversation_structured_buffer"
TAX3PO_API_ENDPOINT_DOC_UPDATE="/doc-update"

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'ftov1!91yf@7f7&g2%*@0_e^)ac&f&9jeloc@#v76#^b1dhbl#'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*', '10.4.39.16']


OKTA_CLIENT = {
    "orgUrl": REACT_APP_OKTA_SERVER_URL,
    "token": REACT_APP_OKTA_CLIENT_ID
}


REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        # 'authentication.okta.okta_authentication.OktaAuthentication',
        # 'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ),
    'DEFAULT_FILTER_BACKENDS': (
        'django_filters.rest_framework.DjangoFilterBackend',
    ),
}

# AUTH_USER_MODEL = 'authentication.models.CustomUser'

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'django_filters',
    'authentication',
    'text_bot',
    'drf_yasg',
    'corsheaders',
]

SITE_ID = 1

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
]

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CORS_ORIGIN_ALLOW_ALL = True

ROOT_URLCONF = 'api_crud.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'api_crud.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/2.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.0/howto/static-files/

STATIC_URL = '/static/'












# # Quick-start development settings - unsuitable for production
# # See https://docs.djangoproject.com/en/4.0/howto/deployment/checklist/

STATICFILES_FINDERS = (
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
    # other finders..
    "compressor.finders.CompressorFinder"
)

CORS_ALLOW_CREDENTIALS = True
CORS_ORIGIN_WHITELIST = []
#CSRF_TRUSTED_ORIGINS = []

# Email

# client: example.com/activate/token
ACTIVATION_PATH_ON_EMAIL = "activate"
PASSWORD_RESET_PATH_ON_EMAIL = "password-reset"
PASSWORD_SET_PATH_ON_EMAIL = "password-set"

# email templates
EMAIL_TEMPLATE_ACTIVATION = "verification"
EMAIL_TEMPLATE_PASSWORD_RESET = "forgot-password"
EMAIL_TEMPLATE_PASSWORD_SET = "password-set"

# email subjects templates
EMAIL_SUBJECT_ACTIVATION = "Verify your email address"
EMAIL_SUBJECT_PASSWORD_RESET = "Forgot Password"
EMAIL_SUBJECT_PASSWORD_SET = "Set your password"

# tokens
EXPIRATION_ACTIVATION_TOKEN = datetime.timedelta(days=7)
EXPIRATION_PASSWORD_RESET_TOKEN = datetime.timedelta(hours=1)
EXPIRATION_PASSWORD_SET_TOKEN = datetime.timedelta(days=7)

EMAIL_TEMPLATE_VARIABLES = {"domain": "localhost:3000"}


SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': datetime.timedelta(minutes=5),
    'REFRESH_TOKEN_LIFETIME': datetime.timedelta(days=1),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
}

# https://docs.djangoproject.com/en/4.2/topics/auth/customizing/#specifying-authentication-backends
# The order of AUTHENTICATION_BACKENDS matters, so if the same username and password is valid in multiple backends,
# Django will stop processing at the first positive match.

LOGIN_URL = "/login"

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.0/howto/static-files/

# STATIC_URL = 'static/'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

STATIC_ROOT = os.path.join(BASE_DIR, "static")

# Default primary key field type
# https://docs.djangoproject.com/en/4.0/ref/settings/#default-auto-field

# DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# use S3
DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"

# CELERY STUFF
CELERY_BROKER_URL = "redis://localhost:6379"
CELERY_RESULT_BACKEND = "redis://localhost:6379"
# CELERY_ACCEPT_CONTENT = ['application/json']
# CELERY_TASK_SERIALIZER = 'json'
# CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = "US/Eastern"
CELERY_TASK_TIME_LIMIT = 3600  # 1 hour
CELERY_TASK_SOFT_TIME_LIMIT = 3600
# CELERY_BEAT_SCHEDULE = {
#    'send-daily-report': {
#        'task': 'ir.tasks.send_daily_report',
#        'schedule': datetime.timedelta(minutes=15),
#    },
# }


CELERY_BEAT_SCHEDULE = {
    "generate_docusign_access_token": {
        "task": "wizard_dashboard.utils.get_access_token",
        "schedule": datetime.timedelta(hours=7),
    },
    "csc_overview_calculations": {
        "task": "application.utils.update_csc_overview",
        "schedule": datetime.timedelta(minutes=30),
    },
    "compiler": {
        "task": "data.utils.compile_data",
        "schedule": datetime.timedelta(minutes=5),
    },
    "flush_expired_tokens": {
        "task": "api_v2.tasks.flush_expired_tokens",
        "schedule": datetime.timedelta(days=1),
    }
}

PHONENUMBER_DEFAULT_FORMAT = "E164"
PHONENUMBER_DB_FORMAT = "E164"
PHONENUMBER_DEFAULT_REGION = 'US'


X_FRAME_OPTIONS = "SAMEORIGIN"

CSRF_TRUSTED_ORIGINS = [
    "https://app.innovationrefunds.com",
    "https://cream.innovationrefunds.com",
    "https://milkshake.innovationrefunds.com",
]

# File Upload Size, 25M
MAX_FILE_SIZE = 25_600_000

API_KEY_INTEGRATION_POUNDCAKE_APP = None

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "normal": {
            "format": "[{levelname}] {asctime} {message}",
            "style": "{"
        }
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "normal"
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": os.getenv("DJANGO_LOG_LEVEL", "INFO"),
        },
        "api_crud": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": True,
        },
        "application": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": True,
        },
        "csc": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": True,
        },
        "manager": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": True,
        },
    },
}

# IR Default URL
IR_URL = "https://www.innovationrefunds.com/"

# Miscellaneous
SUPPORTED_IMAGE_MIMETYPES = ['image/jpeg', 'image/png', 'image/gif']

# THIS HAS TO BE AT END OF FILE
try:
    from api_crud.local_settings import *
except ImportError:
    raise Exception("A local_settings.py file is required to run this project")
