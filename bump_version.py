""" Script to change versioning of files (eg. manifest.yml) for executors
[encoders, crafters, indexers, rankers]. Invoke this script from the
executor directory passing the file name for versioning.
Example usage from TextPaddleHubEncoder executor directory:
python ../../../manifest-version.py "manifest.yml"
Commits the change in the branch and raises a PR for the executor.
"""
import glob
import os
import subprocess

import git
from ruamel.yaml import YAML

repo_dir = os.getcwd()
yaml = YAML()

version_file = 'manifest.yml'
repo = git.Repo()
origin = repo.remote(name='origin')
previous_branch = repo.active_branch

for fpath in glob.glob(f'./**/{version_file}', recursive=True):
    dname = fpath.split('/')[-2]
    with open(fpath) as fp:
        info = yaml.load(fp)
        old_ver = info['version']
        new_ver = '.'.join(old_ver.split('.')[:-1] + [str(int(old_ver.split('.')[-1]) + 3)])
        info['version'] = new_ver
    with open(fpath, 'w') as fp:
        yaml.dump(info, fp)

    br_name = ''
    try:
        print('preparing the branch ...')
        br_name = f'chore-{dname.lower()}-{new_ver.replace(".", "")}'
        new_branch = repo.create_head(br_name)
        new_branch.checkout()

        print(f'bumping version to {new_ver} and committing to {new_branch}...')
        repo.git.add(update=True)
        repo.index.commit(f'chore: bump {dname} version to {new_ver}')
        repo.git.push("--set-upstream", origin, repo.head.ref)

        # make PR using `gh`
        print('making a PR ...')
        title_string = f'bumping version for {dname} to {new_ver}'
        body_string = f'bumping version from {old_ver} to {new_ver}'
        pr_command = f'gh pr create -f'
        subprocess.call(pr_command, shell=True)
    except:
        raise
    finally:
        repo.git.checkout('master')
        if br_name:
            repo.delete_head(br_name)
