""" Script to change versioning of manifest.yml for executors
[encoders, crafters, indexers, rankers]. Invoke this script from the
executor directory passing old version and new version.
Example usage from TextPaddleHubEncoder executor directory:
python ../../../manifest-version.py
"""
import os, fnmatch
old_version = "version: 0.0.1"
latest_version = "version: 0.0.2"
filePattern = "manifest.yml"

for dname, dirs, files in os.walk(os.getcwd()):
    for fname in fnmatch.filter(files, filePattern):
        fpath = os.path.join(dname, fname)
        print(fpath)
        with open(fpath) as f:
            s = f.read()
        s = s.replace(old_version, latest_version)
        with open(fpath, "w") as f:
            f.write(s)
