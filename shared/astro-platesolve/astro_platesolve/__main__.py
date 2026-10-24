"""Explicit central engine configuration; never downloads engines or indexes."""
import argparse,json
from .api import settings_path,read_settings

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('show')
    configure=sub.add_parser('configure')
    configure.add_argument('--backend',choices=['astap','skyshot'],required=True)
    configure.add_argument('--executable',required=True)
    configure.add_argument('--database',default='')
    configure.add_argument('--index',default='')
    args=parser.parse_args()
    path=settings_path()
    if args.command=='show': print(json.dumps({'path':str(path),**read_settings()},indent=2));return
    value={'backend':args.backend,'executable':args.executable,'database':args.database,
           'options':{'index':args.index} if args.index else {}}
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    print(f'Shared solver configured: {args.backend} ({path})')
if __name__=='__main__':main()
