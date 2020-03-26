import json
import os
import pathlib
import subprocess
from datetime import datetime
from pathlib import Path

from builder.app import load_manifest, get_parser

# current date and time
cur_dir = pathlib.Path(__file__).parent.absolute()
builder_files = list(Path(cur_dir).glob('**/*'))
build_hist_path = os.path.join(cur_dir, 'build-history.json')

root_dir = pathlib.Path(__file__).parents[1].absolute()
hub_files = list(Path(root_dir).glob('hub/**/*.y*ml')) + \
            list(Path(root_dir).glob('hub/**/*Dockerfile')) + \
            list(Path(root_dir).glob('hub/**/*.py'))

builder_revision = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()


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


def build_on_update():
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
            break

    update_targets = set()
    for p in hub_files:
        modified_time = get_modified_time(p)
        if modified_time > last_build_time or is_builder_updated:
            target = str(pathlib.Path(str(p)).parent.absolute())
            update_targets.add(target)

    if update_targets:
        for p in update_targets:
            canonic_name = os.path.relpath(p).replace('/', '.')
            try:
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
    else:
        print('noting to build')

    hist = {
        'LastBuildTime': get_now_timestamp(),
        'Images': image_map,
        'BuildStatus': status_map,
        'BuilderRevision': builder_revision
    }

    with open(build_hist_path, 'w') as fp:
        json.dump(hist, fp)


if __name__ == '__main__':
    build_on_update()
