import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path

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
jinasrc_dir = os.path.join(root_dir, 'src', 'jina')
image_tag_regex = r'^hub.[a-zA-Z_$][a-zA-Z_\s\-\.$0-9]*$'
label_prefix = 'ai.jina.hub.'
docker_registry = 'jinaai/'

# current date and time
builder_files = list(Path(root_dir).glob('builder/*'))

build_hist_path = os.path.join(root_dir, 'status', 'build-history.json')
readme_path = os.path.join(root_dir, 'status', 'README.md')
hubbadge_path = os.path.join(root_dir, 'status', 'hub-stat.svg')

hub_files = list(Path(root_dir).glob('hub/**/*.y*ml')) + \
            list(Path(root_dir).glob('hub/**/*Dockerfile')) + \
            list(Path(root_dir).glob('hub/**/*.py'))

builder_revision = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()
build_badge_regex = r'<!-- START_BUILD_BADGE -->(.*)<!-- END_BUILD_BADGE -->'
build_badge_prefix = r'<!-- START_BUILD_BADGE --><!-- END_BUILD_BADGE -->'


def safe_url_name(s):
    return s.replace('-', '--').replace('_', '__').replace(' ', '_')


def get_badge_md(img_name, is_success=True):
    success_tag = 'success-success' if is_success else 'fail-critical'
    return f'[![{img_name}](https://img.shields.io/badge/{safe_url_name(img_name)}-' \
           f'{success_tag}?style=flat-square)]' \
           f'(https://hub.docker.com/repository/docker/jinaai/{img_name})'


def get_now_timestamp():
    now = datetime.now()
    return int(datetime.timestamp(now))


def get_modified_time(p) -> int:
    r = subprocess.check_output(
        ['git', 'log', '-1', '--pretty=%at', str(p)]).strip().decode()
    if r:
        return int(r)
    else:
        print(f'can not fetch the modified time of {p}, is it under git?')
        return 0


def set_reason(args, reason):
    if not args.reason:
        args.reason = [reason]
    else:
        args.reason.append(reason)


def build_multi_targets(args):
    try:
        with open(build_hist_path, 'r') as fp:
            hist = json.load(fp)
            last_build_time = hist.get('LastBuildTime', 0)
            image_map = hist.get('Images', {})
            status_map = hist.get('LastBuildStatus', {})
    except:
        raise ValueError('can not fetch "LastBuildTime" from build-history.json')

    print(f'last build time: {last_build_time}')

    # check if builder is updated
    is_builder_updated = False
    for p in builder_files:
        if get_modified_time(p) > last_build_time:
            print(f'builder is updated '
                  f'because of the modified time of {p} '
                  f'is later than last build time.\n'
                  f'means, I need to rebuild ALL images')
            is_builder_updated = True
            set_reason(args, 'builder is updated need to rebuild all images')
            break

    update_targets = set()
    for p in hub_files:
        modified_time = get_modified_time(p)
        if modified_time > last_build_time or is_builder_updated:
            target = str(pathlib.Path(str(p)).parent.absolute())
            update_targets.add(target)

    if update_targets:
        if not is_builder_updated:
            set_reason(args, f'{update_targets} are updated and need to be rebuilt')
        for p in update_targets:
            canonic_name = os.path.relpath(p).replace('/', '.')
            try:
                args.target = p
                image_name = build_target(args)
                subprocess.check_call(['docker', 'pull', image_name])
                tmp = subprocess.check_output(['docker', 'inspect', image_name]).strip().decode()
                tmp = json.loads(tmp)[0]
                if canonic_name not in image_map:
                    image_map[canonic_name] = []
                image_map[canonic_name].append({
                    'Status': True,
                    'LastBuildTime': get_now_timestamp(),
                    'Inspect': tmp,
                })
                status_map[canonic_name] = True
            except Exception as ex:
                status_map[canonic_name] = False
                print(ex)

        # update readme
        with open(readme_path, 'r') as fp:
            tmp = fp.read()
            badge_str = '\n'.join([get_badge_md(k, v) for k, v in status_map.items()])
            h1 = f'## Last Build at: {datetime.now():%Y-%m-%d %H:%M:%S %Z}'
            h2 = '<summary>Reason</summary>'
            h3 = '**Images**'
            reason = '\n\n'.join([v for v in args.reason])
            tmp = re.sub(pattern=build_badge_regex,
                         repl='\n\n'.join([build_badge_prefix, h1, h3, badge_str,
                                           '<details>', h2, reason, '</details>']),
                         string=tmp, flags=re.DOTALL)

        with open(readme_path, 'w') as fp:
            fp.write(tmp)
    else:
        set_reason(args, f'but i have nothing to build')
        print('noting to build')

    # update json track
    with open(build_hist_path, 'w') as fp:
        json.dump({
            'LastBuildTime': get_now_timestamp(),
            'LastBuildReason': args.reason,
            'LastBuildStatus': status_map,
            'BuilderRevision': builder_revision,
            'Images': image_map,
        }, fp)

    update_hub_badge(len(image_map))
    print('delivery success')


def copy_src_to_context(target_path):
    shutil.copytree(jinasrc_dir, os.path.join(target_path, 'jina'))


def update_hub_badge(img_count):
    try:
        from urllib.request import Request, urlopen
        request_headers = {
            "Accept-Language": "en-US,en;q=0.5",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "http://thewebsite.com",
            "Connection": "keep-alive"
        }
        url = f'https://badgen.net/badge/Hub%20Images/{img_count}/cyan'
        request = Request(url, headers=request_headers)

        with urlopen(request) as d, open(hubbadge_path, 'wb') as opfile:
            data = d.read()
            opfile.write(data)
    except Exception as ex:
        print(ex)
        print('something wrong, badge is not updated')


def remove_control_characters(s):
    return ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C')


def build_target(args):
    if os.path.exists(args.target) and os.path.isdir(args.target):
        dockerfile_path = os.path.join(args.target, 'Dockerfile')
        manifest_path = os.path.join(args.target, 'manifest.yml')
        for j in (dockerfile_path, manifest_path):
            if not os.path.exists(j):
                raise FileNotFoundError(f'{j} does not exist and it is required for a valid hub image!')
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

    copy_src_to_context(args.target)

    dockerbuild_cmd = ['docker', 'buildx', 'build']
    dockerbuild_args = ['--platform', ','.join(v for v in _manifest['platform']),
                        '-t', f'{docker_registry}{image_canonical_name}:{_manifest["version"]}', '-t',
                        f'{docker_registry}{image_canonical_name}:latest',
                        '--file', dockerfile_path + '.tmp']
    dockerbuild_action = '--push' if args.push else '--load'
    docker_cmd = dockerbuild_cmd + dockerbuild_args + [dockerbuild_action, args.target]
    subprocess.check_call(docker_cmd)
    print('build success!')

    if args.push:
        target_readme_path = os.path.join(args.target, 'README.md')
        if not os.path.exists(target_readme_path):
            with open(target_readme_path, 'w') as fp:
                fp.write('#{name}\n\n#{description}\n'.format_map(_manifest))

        docker_readme_cmd = ['docker', 'run', '-v', f'{args.target}:/workspace',
                             '-e', f'DOCKERHUB_USERNAME={os.environ["DOCKERHUB_DEVBOT_USER"]}',
                             '-e', f'DOCKERHUB_PASSWORD={os.environ["DOCKERHUB_DEVBOT_PWD"]}',
                             '-e', f'DOCKERHUB_REPOSITORY={docker_registry}{image_canonical_name}',
                             '-e', 'README_FILEPATH=/workspace/README.md',
                             'peterevans/dockerhub-description:2.1']
        subprocess.check_call(docker_readme_cmd)
        print('upload readme success!')

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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str,
                        help='the directory path of target Pod image, where manifest.yml and Dockerfile located')
    gp1 = parser.add_mutually_exclusive_group()
    gp1.add_argument('--push', action='store_true', default=False,
                     help='push to the registry')
    gp1.add_argument('--test', action='store_true', default=False,
                     help='test the pod image')
    parser.add_argument('--error_on_empty', action='store_true', default=False,
                        help='stop and raise error when the target is empty, otherwise just gracefully exit')
    parser.add_argument('--reason', type=str, nargs='*',
                        help='the reason of the build')
    return parser


if __name__ == '__main__':
    a = get_parser().parse_args()
    if a.target:
        build_target(a)
    else:
        build_multi_targets(a)
