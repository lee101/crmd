from pathlib import Path
from setuptools import setup


def _read_requirements():
    req_path = Path(__file__).with_name("requirements.in")
    return [line.strip() for line in req_path.read_text().splitlines() if line.strip()]


setup(
    name="crmd",
    version="0.2.0",
    py_modules=["crmd"],
    install_requires=_read_requirements(),
    entry_points={"console_scripts": ["crmd=crmd:app", "crmd-worker=crmd:worker_cmd"]},
)
