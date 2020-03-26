import argparse
import json
import os
import pathlib
import re
import subprocess
from datetime import datetime
from pathlib import Path

from builder.app import load_manifest, get_parser

# current date and time
cur_dir = pathlib.Path(__file__).parent.absolute()
builder_files = list(Path(cur_dir).glob('**/*'))
build_hist_path = os.path.join(cur_dir, 'build-history.json')

root_dir = pathlib.Path(__file__).parents[1].absolute()
readme_path = os.path.join(root_dir, 'README.md')
hub_files = list(Path(root_dir).glob('hub/**/*.y*ml')) + \
            list(Path(root_dir).glob('hub/**/*Dockerfile')) + \
            list(Path(root_dir).glob('hub/**/*.py'))

builder_revision = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()
build_badge_regex = r'(?<=<! -- START_BUILD_BADGE -->)(.*)(?=<! -- END_BUILD_BADGE -->)'


def safe_url_name(s):
    return s.replace('-', '--').replace('_', '__').replace(' ', '_')


def get_badge_md(img_name, is_success=True):
    if is_success:
        return f'![{img_name}](https://img.shields.io/badge/{safe_url_name(img_name)}-success-success?style=flat-square)'
    else:
        return f'![{img_name}](https://img.shields.io/badge/{safe_url_name(img_name)}-fail-critical?style=flat-square)'


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


def build_on_update(args):
    try:
        with open(build_hist_path, 'r') as fp:
            hist = json.load(fp)
            last_build_time = hist.get('LastBuildTime', 0)
            image_map = hist.get('Images', {})
            status_map = hist.get('BuildStatus', {})
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
        set_reason(args, f'{update_targets} are updated and need to be rebuilt')
        for p in update_targets:
            canonic_name = os.path.relpath(p).replace('/', '.')
            try:
                if args.push:
                    image_name = load_manifest(get_parser().parse_args([p, '--push']))
                elif args.test:
                    image_name = load_manifest(get_parser().parse_args([p, '--test']))
                else:
                    image_name = load_manifest(get_parser().parse_args([p]))
                tmp = subprocess.check_output(['docker', 'inspect', image_name]).strip().decode()
                tmp = json.loads(tmp)[0]
                image_map[tmp['Id']] = {
                    'Status': True,
                    'LastBuildTime': get_now_timestamp(),
                    'Inspect': tmp,
                    'DisplayName': canonic_name
                }
                status_map[canonic_name] = True
            except Exception as ex:
                status_map[canonic_name] = False
                print(ex)

        # update readme
        with open(readme_path, 'r') as fp:
            tmp = fp.read()
            badge_str = ' '.join([get_badge_md(b) for b in status_map])
            badge_header = f'> Last Build Status: {datetime.now():%Y-%m-%d %H:%M:%S}'
            tmp = re.sub(pattern=build_badge_regex,
                         repl=f'\n\n{badge_header}\n\n{badge_str}\n\n',
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
            'BuildStatus': status_map,
            'BuilderRevision': builder_revision,
            'Images': image_map,
        }, fp)

    print('delivery success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reason', type=str, nargs='*',
                        help='the reason of the build')
    gp1 = parser.add_mutually_exclusive_group()
    gp1.add_argument('--push', action='store_true', default=False,
                     help='push to the registry')
    gp1.add_argument('--test', action='store_true', default=False,
                     help='test the pod image')
    build_on_update(parser.parse_args())
