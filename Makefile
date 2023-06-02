test:
	cargo test

check:
	cargo fix
	make test
	cargo publish --dry-run

publish:
	make check
	cargo publish

patch:
	python3 update_version.py patch

minor:
	python3 update_version.py minor

major:
	python3 update_version.py major