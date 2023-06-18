test:
	cargo test

bench:
	cargo bench

check:
	cargo fix
	make test
	make bench
	cargo publish --dry-run

publish:
	make check
	cargo publish

docu:
	RUSTDOCFLAGS="--html-in-header katex.html" cargo doc --no-deps --open

patch:
	python3 update_version.py patch

minor:
	python3 update_version.py minor

major:
	python3 update_version.py major
