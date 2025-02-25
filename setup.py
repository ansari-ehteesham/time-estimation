import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_descr = f.read()

__version__ = "0.0.0"

REPO_NAME = "time-estimation"
AUTHOR_USER_NAME = "ansari-ehteesham"
SRC_REPO = "timeEstimator"
AUTHOR_EMAIL = "an.ehteesham@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    long_description = long_descr,
    long_description_content = "text/markdown",
    description="Package for Estimating Time from an Analog Clock Image.",
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)