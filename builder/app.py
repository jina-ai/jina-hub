import argparse
import os
import pathlib
import re
import subprocess
import unicodedata

from ruamel.yaml import YAML

yaml = YAML()
allowed = {'name', 'description', 'author', 'url', 'documentation', 'version', 'vendor', 'license', 'avatar',
           'platform'}
required = {'name', 'description'}
sver_regex = r'^(=|>=|<=|=>|=<|>|<|!=|~|~>|\^)?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)' \
             r'\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)' \
             r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+' \
             r'(?:\.[0-9a-zA-Z-]+)*))?$'
name_regex = r'^[a-zA-Z_$][a-zA-Z_\s\-$0-9]{2,20}$'
cur_dir = pathlib.Path(__file__).parent.absolute()
root_dir = pathlib.Path(__file__).parents[1].absolute()
image_tag_regex = r'^hub.[a-zA-Z_$][a-zA-Z_\s\-\.$0-9]*$'
label_prefix = 'ai.jina.hub.'
docker_registry = 'jinaai/'


def remove_control_characters(s):
    return ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C')


def load_manifest(args):
    if os.path.exists(args.target) and os.path.isdir(args.target):
        dockerfile_path = os.path.join(args.target, 'Dockerfile')
        manifest_path = os.path.join(args.target, 'manifest.yml')
        if not os.path.exists(dockerfile_path):
            raise FileNotFoundError(f'{dockerfile_path} does not exist!')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f'{manifest_path} does not exist!')
    else:
        if args.error_on_empty:
            raise NotADirectoryError(f'{args.target} is not a valid directory')
        else:
            return

    image_canonical_name = os.path.relpath(args.target).replace('/', '.')
    check_image_name(image_canonical_name)

    with open(os.path.join(cur_dir, 'manifest.yml')) as fp:
        _manifest = yaml.load(fp)  # type: dict
    with open(manifest_path) as fp:
        _manifest.update(yaml.load(fp))

    # check if all keys are allowed keys
    for k in _manifest.keys():
        if k not in allowed:
            raise ValueError(f'{k} is not allowed as a key in manifest.yml, only one of the {allowed}')

    # check the required field in manifest
    for r in required:
        if r not in _manifest:
            raise ValueError(f'{r} is missing in the manifest.yaml, it is required')

    # check if all fields are there
    for r in allowed:
        if r not in _manifest:
            print(f'{r} is missing in your manifest.yml, you may want to check it')

    # replace all chars in value to safe chars
    for k, v in _manifest.items():
        if v and isinstance(v, str):
            _manifest[k] = remove_control_characters(v)

    # check name
    check_name(_manifest['name'])
    # check version number
    check_version(_manifest['version'])
    # check version number
    check_license(_manifest['license'])
    # add revision
    add_revision_source(_manifest)
    # check platform
    if not isinstance(_manifest['platform'], list):
        _manifest['platform'] = list(_manifest['platform'])
    check_platform(_manifest['platform'])

    # show manifest key-values
    for k, v in _manifest.items():
        print(f'{k}: {v}')

    # modify dockerfile
    revised_dockerfile = []
    with open(dockerfile_path) as fp:
        for l in fp:
            revised_dockerfile.append(l)
            if l.startswith('FROM'):
                revised_dockerfile.append('LABEL ')
                revised_dockerfile.append(' \\      \n'.join(f'{label_prefix}{k}="{v}"' for k, v in _manifest.items()))
    for k in revised_dockerfile:
        print(k)

    with open(dockerfile_path + '.tmp', 'w') as fp:
        fp.writelines(revised_dockerfile)

    dockerbuild_cmd = ['docker', 'buildx', 'build']
    dockerbuild_args = ['--platform', ','.join(v for v in _manifest['platform']),
                        '-t', f'{docker_registry}{image_canonical_name}:{_manifest["version"]}', '-t',
                        f'{docker_registry}{image_canonical_name}:latest',
                        '--file', dockerfile_path + '.tmp']
    dockerbuild_action = '--push' if args.push else '--load'
    docker_cmd = dockerbuild_cmd + dockerbuild_args + [dockerbuild_action, args.target]
    subprocess.check_call(docker_cmd)
    print('build success!')
    img_name = f'{docker_registry}{image_canonical_name}:{_manifest["version"]}'

    if args.test:
        test_docker_cli(img_name)
        test_jina_cli(img_name)
        test_flow_api(img_name)

        print('all tests success!')

    return img_name

def test_docker_cli(img_name):
    print('testing image with docker run')
    subprocess.check_call(['docker', 'run', '--rm', img_name, '--max_idle_time', '5', '--shutdown_idle'])


def test_jina_cli(img_name):
    print('testing image with jina cli')
    subprocess.check_call(['jina', 'pod', '--image', img_name, '--max_idle_time', '5', '--shutdown_idle'])


def test_flow_api(img_name):
    print('testing image with jina flow API')
    from jina.flow import Flow
    with Flow().add(image=img_name, replicas=3).build():
        pass


def check_image_name(s):
    if not re.match(image_tag_regex, s):
        raise ValueError(f'{s} is not a valid image name for a Jina Hub image, it should match with {image_tag_regex}')


def check_platform(s):
    with open(os.path.join(cur_dir, 'platforms.yml')) as fp:
        platforms = yaml.load(fp)

    for ss in s:
        if ss not in platforms:
            raise ValueError(f'platform {ss} is not supported, should be one of {platforms}')


def check_license(s):
    with open(os.path.join(cur_dir, 'osi-approved.yml')) as fp:
        approved = yaml.load(fp)
    if s not in approved:
        raise ValueError(f'license {s} is not an OSI-approved license {approved}')
    return approved[s]


def add_revision_source(d):
    d['revision'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()
    d['source'] = 'https://github.com/jina-ai/jina-hub/commit/' + d['revision']


def check_name(s):
    if not re.match(name_regex, s):
        raise ValueError(f'{s} is not a valid name, it should match with {name_regex}')


def check_version(s):
    if not re.match(sver_regex, s):
        raise ValueError(f'{s} is not a valid semantic version number, see http://semver.org/')


def find_potential_targets():
    pass


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str,
                        help='the directory path of target Pod image, where manifest.yml and Dockerfile located')
    gp1 = parser.add_mutually_exclusive_group()
    gp1.add_argument('--push', action='store_true', default=False,
                     help='push to the registry')
    gp1.add_argument('--test', action='store_true', default=False,
                     help='test the pod image')
    parser.add_argument('--error_on_empty', action='store_true', default=False,
                        help='stop and raise error when the target is empty, otherwise just gracefully exit')

    return parser


if __name__ == '__main__':
    load_manifest(get_parser().parse_args())
