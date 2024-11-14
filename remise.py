#
# Script de remise
#

import os
import sys
import glob
import zipfile
import datetime
import argparse

print("### DEBUT SCRIPT REMISE ###")

parser = argparse.ArgumentParser()
parser.add_argument("--destdir", help="repertoire de destination", default=os.curdir)
parser.add_argument("prefix", help="archive prefix (nom de projet)")
args = parser.parse_args()

prefix = args.prefix
destdir = args.destdir
basedir = os.path.abspath(os.path.dirname(__file__))

print(f"prefix : {prefix}")
print(f"basedir: {basedir}")
print(f"destdir: {destdir}")

if not os.path.isdir(destdir):
    print("ERREUR: repertoire de destination inexistant")
    sys.exit(1)

loc = [
    "3rdparty/**",
    "src/**",
    "test/**",
    "data/**",
    "CMakeLists.txt",
    "env.sh.in",
    "remise.py",
    "tp.h.in",
    "README.md",
    "*.cmake",
    ".clang-format",
    ".gitignore",
]

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
basename = f"{prefix}-{timestamp}"
archive_name = f"{basename}.zip"
archive_path = os.path.join(destdir, archive_name)

count = 0
with zipfile.ZipFile(archive_path, "w") as fd:
    for item in loc:
        for realname in glob.iglob(item, root_dir=basedir, recursive=True):
            fullname = os.path.join(basedir, realname)
            if os.path.isfile(fullname):
                count += 1
                destname = os.path.join(basename, realname)
                fd.write(fullname, destname)


print(f"nombre de fichiers archivés: {count}")
print(f"archive de remise: {archive_name}")
print("### FIN SCRIPT REMISE ###")
