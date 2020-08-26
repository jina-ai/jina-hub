""" Script to change versioning of files (eg. manifest.yml) for executors
[encoders, crafters, indexers, rankers]. Invoke this script from the
executor directory passing the file name for versioning.
Example usage from TextPaddleHubEncoder executor directory:
python ../../../manifest-version.py "manifest.yml"
Commits the change in the branch and raises a PR for the executor.
"""
import os
import git
import glob
import subprocess
import sys
from ruamel.yaml import YAML

repo_dir = os.getcwd()
yaml = YAML()

version_file = sys.argv[1]
repo = git.Repo.init(repo_dir)

for fpath in glob.glob(f'**/{version_file}'):
	dname = fpath.split("/")[-2]
	with open(fpath) as fp:
		info = yaml.load(fp)
		old_ver = info['version']
		new_ver = ''.join(str(int(old_ver.split('.')[-1])+1) + old_ver.split('.')[:-1])
	new_branch = repo.create_head('chore-version-' + dname)
	new_branch.checkout()
	index = repo.index
	index.add(fpath)
	index.commit("chore: bump version")
	repo.git.push('--set-upstream', 'origin', new_branch)
	title_string = f'Upgrading manifest version for {dname}'
	body_string = "Bumping version"
	pr_command = f'gh pr create --title \"{title_string}\" --body \"{body_string}\"'
	# Command with shell expansion
	subprocess.call(pr_command, shell=True)
	repo.git.checkout('master')
