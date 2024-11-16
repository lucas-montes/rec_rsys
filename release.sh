#!/bin/bash

set -e

msg="$1"
new_version="$2"

update_version() {
    version="${1#v}"
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Invalid version format: $1"
        exit 1
    fi

    current_version=$(grep -oP '(?<=^version = ")[^"$]*' Cargo.toml)

    if [[ "$current_version" == "$version" ]]; then
        echo "Are you sure about that? The same version? Current: $current_version New: $version"
        exit 1
    fi

    sed -i "s/^version = \".*\"$/version = \"$version\"/" Cargo.toml
    echo "Let's go $version"
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 vX.X.X"
    exit 1
fi

cargo fmt --all
cargo clippy --features default -- -Dclippy::all -D warnings

update_version "$new_version"

git add --all
git commit --all --signoff --message "$msg"
git tag -a "$version" -m "Version $version"
git push origin "$version"

