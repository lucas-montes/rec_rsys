import sys
import configparser


def update_version(current_version: str, version_type: str) -> str:
    major, minor, patch = map(int, current_version.split("."))
    if version_type == "patch":
        patch += 1
    elif version_type == "minor":
        minor += 1
        patch = 0
    elif version_type == "major":
        major += 1
        minor = 0
        patch = 0
    return f'"{major}.{minor}.{patch}"'


def set_version(version_type: str) -> None:
    config = configparser.ConfigParser()
    config.read("Cargo.toml")
    current_version = config.get("package", "version").replace('"', "")
    new_version = update_version(current_version, version_type)
    config.set("package", "version", new_version)
    with open("Cargo.toml", "w") as config_file:
        config.write(config_file)
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_version.py [patch | minor | major]")
        sys.exit(1)
    set_version(sys.argv[1])
