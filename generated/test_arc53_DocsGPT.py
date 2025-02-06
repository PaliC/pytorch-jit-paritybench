import sys
_module = sys.modules[__name__]
del sys
application = _module
api = _module
answer = _module
routes = _module
internal = _module
user = _module
tasks = _module
app = _module
cache = _module
celery_init = _module
celeryconfig = _module
core = _module
logging_config = _module
mongo_db = _module
settings = _module
error = _module
extensions = _module
llm = _module
anthropic = _module
base = _module
docsgpt_provider = _module
google_ai = _module
groq = _module
huggingface = _module
llama_cpp = _module
llm_creator = _module
openai = _module
premai = _module
sagemaker = _module
parser = _module
chunking = _module
embedding_pipeline = _module
file = _module
base_parser = _module
bulk = _module
docs_parser = _module
epub_parser = _module
html_parser = _module
image_parser = _module
json_parser = _module
markdown_parser = _module
openapi3_parser = _module
pptx_parser = _module
rst_parser = _module
tabular_parser = _module
crawler_loader = _module
crawler_markdown = _module
github_loader = _module
reddit_loader = _module
remote_creator = _module
sitemap_loader = _module
telegram = _module
web_loader = _module
schema = _module
retriever = _module
brave_search = _module
classic_rag = _module
duckduck_search = _module
retriever_creator = _module
agent = _module
cryptoprice = _module
postgres = _module
llm_handler = _module
tool_action_parser = _module
tool_manager = _module
elevenlabs = _module
google_tts = _module
usage = _module
utils = _module
vectorstore = _module
document_class = _module
elasticsearch = _module
faiss = _module
lancedb = _module
milvus = _module
mongodb = _module
qdrant = _module
vector_creator = _module
worker = _module
wsgi = _module
chatwoot = _module
discord = _module
bot = _module
scripts = _module
code_docs_gen = _module
ingest = _module
migrate_to_v1_vectorstore = _module
old = _module
ingest_rst = _module
ingest_rst_sphinx = _module
java2doc = _module
js2doc = _module
open_ai_func = _module
py2doc = _module
token_func = _module
test_anthropic = _module
test_openai = _module
test_sagemaker = _module
test_app = _module
test_cache = _module
test_celery = _module
test_error = _module
test_openapi3parser = _module
test_vector_store = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps

