check:
	cargo test
	cargo bench
	cargo clippy --all-targets -- -D warnings
	cargo fmt --all -- --check
	cargo publish --dry-run --allow-dirty

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
