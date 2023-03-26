#!/usr/bin/env python3

import argparse
from doccano_client import DoccanoClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Doccano URL")
    parser.add_argument("project_id", help="Project ID")
    parser.add_argument("-U", "--username", required=True, help="Username")
    parser.add_argument("-P", "--password", required=True, help="Password")
    args = parser.parse_args()

    client = DoccanoClient(args.url)
    client.login(username=args.username, password=args.password)

    for example in client.list_examples(args.project_id, False):
        if example.meta.get('verified'):
            client.update_example_state(args.project_id, example.id)
            print(f"Set #{example.id} to done")
        else:
            break


if __name__ == "__main__":
    main()
