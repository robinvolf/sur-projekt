# Klasifikátor hlasu

### Jak přeložit a spustit

1. Nainstalujte si toolchain jazyka Rust - https://www.rust-lang.org/tools/install
2. Nainstalujte si openblas pomocí vašeho balíčkovacího SW (tento program dynamicky linkuje openblas, kvůli funkcím s lineární algebrou, tudíž se bez něj nepřeloží)
2. Program přeložíte pomocí `cargo build --release`
  - Přepínač `--release` zapne optimalizace
3. Program spustíte pomocí `cargo run --release`
  - Help k programu lze vypsat pomocí `cargo run --release -- --help`

### Jak natrénovat
V kořenovém adresáři ZIP archivu spusťte `./download_and_extract_data.sh`, poté v adresáři s tímto souborem spusťte `./prepare_data.sh`, tyto dva skripty stáhnou a připraví data do vhodné adresářové struktury.
Poté již stačí spustit `cargo run --release -- train --gaussians X --em-iters X --regularization X training_data/ trained_model.ron`.
Pro nápovědu, co znamenají jednotlivé parametry spusťte `cargo run --release -- help train`.
