""" Script to change versioning of files (eg. manifest.yml) for executors
[encoders, crafters, indexers, rankers]. Invoke this script from the
executor directory passing the file name for versioning with its old version and new version.
Example usage from TextPaddleHubEncoder executor directory:
python ../../../manifest-version.py "manifest.yml" "version: 0.0.1" "version: 0.0.2"
Commits the change in the branch and raises a PR for the executor.
"""
import os, fnmatch
import git
import subprocess
import sys
repo_dir = os.getcwd()

version_file = sys.argv[1]
old_version = sys.argv[2]
latest_version = sys.argv[3]
repo = git.Repo.init(repo_dir)

for dpath, dirs, files in os.walk(os.getcwd()):
    for fname in fnmatch.filter(files, version_file):
        fpath = os.path.join(dpath, fname)
        dname = dpath.split("/")[-1]
        print(fpath)
		with open(fpath) as f:
		    s = f.read()
		s = s.replace(old_version, latest_version)
		with open(fpath, "w") as f:
		    f.write(s)
		new_branch = repo.create_head('chore-version-' + dname)
		new_branch.checkout()
		index = repo.index
		index.add(fpath)
		index.commit("chore: bump version")
		repo.git.push('--set-upstream', 'origin', new_branch)
		title_string = "Upgrading manifest version for " + dname
		body_string = "Changing version from 0.0.1 to 0.0.2"
		pr_command = "gh pr create --title \"" + title_string + "\" --body \"" + body_string + "\""
		print(pr_command)
		# Command with shell expansion
		subprocess.call(pr_command, shell=True)
		repo.git.checkout('master')
