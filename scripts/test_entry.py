import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from sales_forecasting.utils import hello
print(hello())
# print("=== entry check ===")
# print("python executable:", sys.executable)
# print("cwd:", Path.cwd())
